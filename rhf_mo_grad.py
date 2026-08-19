from pyscf import gto, scf, ao2mo
from pyscf.scf import cphf
import numpy as np
from gradient_integrals import (hcore_generator, overlap_generator,
                                eri_generator, grad_nuc)
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial
import scipy
import math
from openfermion.linalg import get_sparse_operator
import openfermion
from openfermion.ops.operators.qubit_operator import QubitOperator
#import numpy as np
from openfermion.chem import MolecularData
from pyscf_helper import get_hamiltonian_ferm_op_from_mo_ints,get_active_space_effective_mo_integrals
import functools
from openfermion.chem.molecular_data import spinorb_from_spatial
import os
from functools import partial
MAXMEM = float(os.getenv("MAXMEM",2))
np.einsum = partial(np.einsum, optimize=["greedy",1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)

def gradient_mo(mol, mo_coeffs, hcore_mo, tei_mo,rdm1,rdm2, mf,lag,mc, Cphf: bool=True,
        rotation: bool=False):
    """
    Obtain the gradient given that stationarity has been obtained

    :param mol: pyscf.Mole object for getting AO integrals



    :param mo_coeffs: AO-to-MO molecular orbital coefficients
    :param hcore_mo: Core-MO matrix (n-spatial dim x n-spatial dim)
    :param tei_mo: ERI-MO tensor (n-spatial, n-spatial, n-spatial, n-spatial)
    :param opdm:  one-RDM (in SPIN ORBITALS openFermion ordering)
    :param tpdm: two-RDM (in SPIN ORBITALS openFermion ordering)
    :return: N x 3 matrix where N is the number of atoms. Each row is the force
             vector.
    """
    hcore_deriv = hcore_generator(mol)
    ovrlp_deriv = overlap_generator(mol)
    eri_deriv = eri_generator(mol)
    atmlst = range(mol.natm)
    de = np.zeros((len(atmlst), 3))
    nelec = sum(mol.nelec)
    norbs = mo_coeffs.shape[1]
    zero_oei = np.zeros_like(hcore_mo)
    
    for k, ia in enumerate(atmlst):
        h1ao = hcore_deriv(ia)
        s1ao = ovrlp_deriv(ia)
        eriao = eri_deriv(ia)
        h1mo = np.zeros_like(h1ao)
        s1mo = np.zeros_like(s1ao)
        erimo = np.zeros_like(eriao)
        aoslices = mol.aoslice_by_atom()
        shl0, shl1, p0, p1 = aoslices[ia]
        h1mo[0] = of.general_basis_change(h1ao[0], mo_coeffs, key=(1, 0))
        # Y-Core-MO - Hellmann-Feynman term
        h1mo[1] = of.general_basis_change(h1ao[1], mo_coeffs, key=(1, 0))
        # Z-Core-MO - Hellmann-Feynman term
        h1mo[2] = of.general_basis_change(h1ao[2], mo_coeffs, key=(1, 0))

        erimo[0] -= of.general_basis_change(eriao[0], mo_coeffs, key=(1, 0,1, 0)).transpose((0,2,3,1))
        erimo[1] -= of.general_basis_change(eriao[1], mo_coeffs, key=(1,  0,1, 0)).transpose((0,2,3,1))
        erimo[2] -= of.general_basis_change(eriao[2], mo_coeffs, key=(1, 0,1, 0)).transpose((0,2,3,1))
        core_correction=core_correction2=core_correction3=0
        s1mo[0] = of.general_basis_change(s1ao[0], mo_coeffs, key=(1, 0))
        # Y-S-MO
        s1mo[1] = of.general_basis_change(s1ao[1], mo_coeffs, key=(1, 0))
        # Z-S-MO
        s1mo[2] = of.general_basis_change(s1ao[2], mo_coeffs, key=(1, 0))

        nocc = sum(mol.nelec)//2
        nao = mol.nao
        ncore = mc.ncore
        ncas = mc.ncas
        nvir = nao - ncas - ncore
        nuo = nao - nocc
       # print("nocc:",nocc)
       # print("nao:",nao)
       # print("nuo:",nuo)
        natm=mf.mol.natm
        D=2*np.dot(mo_coeffs[:,:nocc],mo_coeffs[:,:nocc].T)
        eri0_ao = mol.intor("int2e")
        A_0_ao = 4 * eri0_ao - eri0_ao.swapaxes(-2, -3) - eri0_ao.swapaxes(-1, -3)
        A_0_mo = np.einsum("up, vq, uvkl, kr, ls -> pqrs", mo_coeffs, mo_coeffs, A_0_ao, mo_coeffs, mo_coeffs)

        def fvind_oo(x):
          A_0_mo_oo = A_0_mo[:nocc, :nocc, nocc:nao, :nocc]
          AX=np.einsum("pqrs, Ars -> Apq", A_0_mo_oo, x)
          return AX.reshape(3, nocc, nocc)
        def fvind_uu(x):
          A_0_mo_uu = A_0_mo[nocc:nao, nocc:nao, nocc:nao, :nocc] 
          AX=np.einsum("pqrs, Ars -> Apq", A_0_mo_uu, x)
          return AX.reshape(3, nuo, nuo)

        def get_u():  # solving the cphf equations to get unocc-occ U matrix
                      # CPHF: (epsilon_p-epsilon_q)*U_pq - U_uo * A = B   This equation is derived by differentiating the FC = SCE equation. where the nondiagonal matrix elements of Fock matrix is zero.
         # eriao = np.transpose(eriao,(0,2,1,3))
          F_1_ao = (h1ao+
                   -np.einsum("Auvkl, kl -> Auv", eriao, D)
                   + 0.5 * np.einsum("Aukvl, kl -> Auv", eriao, D))
          F_1_mo = np.einsum("up, Auv, vq -> Apq", mo_coeffs, F_1_ao, mo_coeffs)
          def fvind(x):
           # AX=np.einsum("pqrs, Ars -> Apq", A_0_mo, x)
            A_0_mo_pqoo = A_0_mo[:, :,:nocc, :nocc].reshape(nao,nao,nocc,nocc)
            AX=np.einsum("pqrs, Ars -> Apq", A_0_mo_pqoo, x)
            return AX
          def fvind_uo(x):
              n1=x.shape[-1]
              n2=x.shape[-2]
              if x.shape[0] != 3:
                  new_x = np.zeros((3,n2,n1), dtype=x.dtype)
                  new_x[:x.shape[0]] = x
                  x=new_x

              A_0_mo_uo = A_0_mo[nocc:nao, :nocc, nocc:nao, :nocc]
              AX=np.einsum("pqrs, Ars -> Apq", A_0_mo_uo, x)
             # print('AX=',AX.shape)
              return AX.reshape(3, nuo, nocc)
          B_1 = np.zeros((3,nao,nao)) 
          B_1 = (
                  + F_1_mo
                      + np.einsum("Apq, q -> Apq", s1mo, mf.mo_energy)  # the sign of s1mo should be taken into consideration(other packages may be different)
                      + 0.5 *fvind(s1mo[ :, :nocc, :nocc])
                          )
          u = cphf.solve(fvind_uo, mf.mo_energy, mf.mo_occ, B_1[ :, nocc:nao, :nocc].reshape(3, nuo, nocc), max_cycle=1000)[0] 
          return u, B_1

        if Cphf is True:
            
          if rotation is True :   # optimizing the molecular orbital coefficients
              u,B_1 = get_u()

              U_1_pq = - 0.5 * s1mo
              U_1_pq[:, nocc:nao, :nocc] = -u
              U_1_pq[:, :nocc, nocc:nao] =  -s1mo[:, :nocc, nocc:nao] - U_1_pq[:, nocc:nao, :nocc].swapaxes(-1, -2)  # using the orthonormality of the molecular orbital
          else:
                u,B_1 = get_u()
                D_pq =  -(mf.mo_energy[:, None] - mf.mo_energy[None, :]) + 1e-300
                U_1_pq = np.zeros((3, nao, nao))
                U_1_pq[:,  nocc:nao,:nocc] = -u
                U_1_pq[:, :nocc,nocc:nao] =  (-s1mo[:,nocc:nao,:nocc] - U_1_pq[:,  nocc:nao,:nocc]).swapaxes(-1, -2)
                U_1_pq[:, :nocc, :nocc] = -((fvind_oo(u)+B_1[:,:nocc, :nocc])/D_pq[:nocc, :nocc])
                U_1_pq[:, nocc:nao, nocc:nao] = -((fvind_uu(u)+B_1[:,nocc:nao, nocc:nao])/D_pq[nocc:nao, nocc:nao])
                for p in range(nao):
                    U_1_pq[:, p, p] = - s1mo[:, p, p] / 2
          I_sum = lag + lag.T
          I_minus = lag - lag.T
          U_minus = U_1_pq - U_1_pq.swapaxes(-1, -2)
          de[k][0] -= 0.5*(np.einsum('pq,pq',U_minus[0,ncore:nao,:ncore],I_minus[ncore:nao,:ncore])+
                          np.einsum('pq,pq',U_minus[0,:ncore,ncore:nao],I_minus[:ncore,ncore:nao])+
                          np.einsum('pq,pq',U_minus[0,ncore+ncas:nao,ncore:ncore+ncas],I_minus[ncore+ncas:nao,ncore:ncore+ncas])+                                       np.einsum('pq,pq',U_minus[0,ncore:ncore+ncas,ncore+ncas:nao],I_minus[ncore:ncore+ncas,ncore+ncas:nao]))
          de[k][1] -= 0.5*(np.einsum('pq,pq',U_minus[1,ncore:nao,:ncore],I_minus[ncore:nao,:ncore])+
                          np.einsum('pq,pq',U_minus[1,:ncore,ncore:nao],I_minus[:ncore,ncore:nao])+
                          np.einsum('pq,pq',U_minus[1,ncore+ncas:nao,ncore:ncore+ncas],I_minus[ncore+ncas:nao,ncore:ncore+ncas])+                                       np.einsum('pq,pq',U_minus[1,ncore:ncore+ncas,ncore+ncas:nao],I_minus[ncore:ncore+ncas,ncore+ncas:nao])) 
          de[k][2] -= 0.5*(np.einsum('pq,pq',U_minus[2,ncore:nao,:ncore],I_minus[ncore:nao,:ncore])+
                          np.einsum('pq,pq',U_minus[2,:ncore,ncore:nao],I_minus[:ncore,ncore:nao])+
                          np.einsum('pq,pq',U_minus[2,ncore+ncas:nao,ncore:ncore+ncas],I_minus[ncore+ncas:nao,ncore:ncore+ncas])+
                          np.einsum('pq,pq',U_minus[2,ncore:ncore+ncas,ncore+ncas:nao],I_minus[ncore:ncore+ncas,ncore+ncas:nao]))
          de[k][0] -= 0.5*np.einsum('pq,pq',s1mo[0],I_sum)
          de[k][1] -= 0.5*np.einsum('pq,pq',s1mo[1],I_sum)
          de[k][2] -= 0.5*np.einsum('pq,pq',s1mo[2],I_sum)
          
        else:
          h1mo[0] += 0.5 * (np.einsum('pj,ip->ij', hcore_mo, s1mo[0]) +
                     np.einsum('ip,jp->ij', hcore_mo, s1mo[0]))
          h1mo[1] += 0.5 * (np.einsum('pj,ip->ij', hcore_mo, s1mo[1]) +
                     np.einsum('ip,jp->ij', hcore_mo, s1mo[1]))
          h1mo[2] += 0.5 * (np.einsum('pj,ip->ij', hcore_mo, s1mo[2]) +
                     np.einsum('ip,jp->ij', hcore_mo, s1mo[2]))
          erimo[0] += 0.5 * (np.einsum('px,xqrs', s1mo[0], tei_mo) +
                     np.einsum('qx,pxrs', s1mo[0], tei_mo) + 
                     np.einsum('rx,pqxs', s1mo[0], tei_mo) +
                     np.einsum('sx,pqrx', s1mo[0], tei_mo))
          erimo[1] += 0.5 * (np.einsum('px,xqrs', s1mo[1], tei_mo) +
                     np.einsum('qx,pxrs', s1mo[1], tei_mo) +                              
                     np.einsum('rx,pqxs', s1mo[1], tei_mo) +                                      
                     np.einsum('sx,pqrx', s1mo[1], tei_mo))
          erimo[2] += 0.5 * (np.einsum('px,xqrs', s1mo[2], tei_mo) +
                     np.einsum('qx,pxrs', s1mo[2], tei_mo) +                              
                     np.einsum('rx,pqxs', s1mo[2], tei_mo) +                                      
                     np.einsum('sx,pqrx', s1mo[2], tei_mo))
          

        de[k][0]+= np.einsum("pq,pq",h1mo[0],rdm1)
        de[k][1]+= np.einsum("pq,pq",h1mo[1],rdm1)
        de[k][2]+= np.einsum("pq,pq",h1mo[2],rdm1)
        de[k][0]+= 0.5*np.einsum("pqrs,pqrs",erimo[0],rdm2)
        de[k][1]+= 0.5*np.einsum("pqrs,pqrs",erimo[1],rdm2)
        de[k][2]+= 0.5*np.einsum("pqrs,pqrs",erimo[2],rdm2)
    de += grad_nuc(mol, atmlst=atmlst)
    return de
