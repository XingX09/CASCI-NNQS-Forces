from rhf_mo_grad import gradient_mo
import openfermion
import numpy
import numpy as np
from pyscf import gto,scf,cc,ci,mp,lo,fci,mcscf
import pyscf_helper
from openfermion.chem.molecular_data import spinorb_from_spatial
import scipy
from rdm import make_rdm12,get_rdm1_from_casdm1, get_rdm2_from_casdm
from scipy.sparse import csr_matrix
from datetime import datetime
def test_rdm_molecule(mol,mol_name,n_qubits,nact):

    data_mo_coeff = np.load('mo_coeff.npz')
    mo_coeff = data_mo_coeff['mo_coeff']
    oei, tei = pyscf_helper.get_mo_integrals_from_molecule_and_hf_orb(mol, mo_coeff)


    data = np.load("ci_psis.npz")
    n_samples, samples, psis = data['n_samples'], data[ 'samples'], data['psis']
    vec_length = 2**n_qubits

    idx_list = []
    for i in range(len(psis)):
        HFstate = samples[(i+0)*n_qubits:(i+1)*n_qubits]
        idx = ""
        for j in HFstate:
          if j ==-1:
              idx+="0"
          if j == 1:
              idx+="1"
        idx = int(idx,2)
        idx_list.append(idx)
   # print(type(idx_list))
    strings=np.array(idx_list, dtype=np.int64) 
    n_orb = nact
    rdm1_mo,rdm2_mo = make_rdm12(psis,strings,n_orb)
    rdm1_mo=rdm1_mo[::-1,::-1]
    rdm2_mo=rdm2_mo[::-1,::-1,::-1,::-1]
    print("rdm1_mo 维度:", rdm1_mo.shape)

    return rdm1_mo,rdm2_mo,oei,tei,mo_coeff

def get_ful_rdm(rdm1,rdm2,n_qubits,n_fro,n_vir):
  n_tot=n_qubits+n_fro+n_vir
  rdm1_ful=numpy.zeros([n_tot] * 2, dtype=numpy.complex128)
  rdm2_ful = numpy.zeros([n_tot] * 4, dtype=numpy.complex128)
  for i in range(n_fro):
    rdm1_ful[i,i]=2
    
    for j in range(n_fro):
      if not i==j: 
        rdm2_ful[i,i,j,j]=-2
        rdm2_ful[i,j,j,i]=2

  for i in range(n_fro):
    for p in range(n_fro,n_qubits+n_fro):
        
      for q in range(n_fro,n_qubits+n_fro):
        rdm1_ful[p,q]=rdm1[p-n_fro,q-n_fro]
        rdm2_ful[i,i,p,q]=-rdm1[p-n_fro,q-n_fro]
        rdm2_ful[p,i,i,q]=rdm1[p-n_fro,q-n_fro]
        rdm2_ful[i,p,i,q]=-rdm1[p-n_fro,q-n_fro]
        rdm2_ful[p,i,q,i]=rdm1[p-n_fro,q-n_fro]
        rdm2_ful[q,p,i,i]=-rdm1[p-n_fro,q-n_fro]
        rdm2_ful[i,q,p,i]=rdm1[p-n_fro,q-n_fro]

  for p in range(n_fro,n_qubits+n_fro):
    for q in range(n_fro,n_qubits+n_fro):    
      for u in range(n_fro,n_qubits+n_fro):
           # v=m
            for v in range(n_fro,n_qubits+n_fro):
              rdm2_ful[p,q,u,v]=rdm2[p-n_fro,q-n_fro,u-n_fro,v-n_fro]
  return rdm1_ful,rdm2_ful

def get_two_rdm_from_spin(rdm_spin,norbs):
    rdm2 = np.zeros((norbs,norbs,norbs,norbs))
    for p in range(norbs):
        for q in range(norbs):
            for r in range(norbs):
                for s in range(norbs):
                    rdm2[p][q][r][s] = -(rdm_spin[2*p][2*q][2*r+1][2*s+1]+rdm_spin[2*p+1][2*q+1][2*r+1][2*s+1]+rdm_spin[2*p][2*q][2*r][2*s]+rdm_spin[2*p+1][2*q+1][2*r][2*s]-rdm_spin[2*p][2*q+1][2*r+1][2*s]+rdm_spin[2*p+1][2*q][2*r][2*s+1]+rdm_spin[2*p+1][2*q][2*r+1][2*s]+rdm_spin[2*p][2*q+1][2*r][2*s+1])
    return rdm2

if __name__ == "__main__":
    mol=gto.Mole()
   # mol.atom=[('O',(0, 0, 0)), ('Li',(-r, 0, 0)), ('Li',(r, 0, 0))]
   # mol.atom=[['Be',(0,0,0)],['H',(0,0,r)],['H',(0,0,-r)]]
   # mol.atom=[['N',(-0.6,0,0)],['N',(0.6,0,0)]]
    mol.atom=[['Li',(0,0,0)],['F',(1.4,0,0)]]
  #  mol.atom=[['Li',(0,0,0)],['H',(r,0,0)]]
                                                                                   
    #bond_angle=104.5
    #r2=r=1.0
    #x1=r*numpy.sin((bond_angle/2)*(numpy.pi/180))
    #y1=r*numpy.cos((bond_angle/2)*(numpy.pi/180))
    #x2=-r2*numpy.sin((bond_angle/2)*(numpy.pi/180))
    #y2=r2*numpy.cos((bond_angle/2)*(numpy.pi/180))        
    #mol.atom=[('O',(0,0,0)),('H',(x1,y1,0)),('H',(x2,y2,0))]

    mol.basis='6-31g(d)'
    mol.spin=0
    mol.charge=0
    mol.build()
    mf=scf.RHF(mol)
    mf.kernel()
    nelec = sum(mol.nelec)
    norbs = mf.mo_coeff.shape[1]

    mol_name = 'LiF'
    fro_mo_list = [0,1]
    act_mo_list = [2,3,4,5,6,7,8,9]
    Fro = True
    mymc = mcscf.CASCI(mf,8,8)
    ncore_mo = len(fro_mo_list)
    nact = len(act_mo_list)
    n_qubit=len(act_mo_list)*2

    rdm1_mo,rdm2_mo,oei,tei,mo_coeff=test_rdm_molecule(mol,mol_name,n_qubit,nact)
    np.set_printoptions(threshold=np.inf)
  #  print("rdm2_mo:",rdm2_mo)
   # print("rdm1_mo",rdm1_mo)
    n_fro=len(fro_mo_list)

    if Fro is True:
        rdm2_mo = get_rdm2_from_casdm(rdm1_mo, rdm2_mo, ncore_mo, nact, norbs)
        rdm1_mo = get_rdm1_from_casdm1(rdm1_mo, ncore_mo, nact, norbs)

    rdm2_mo = (rdm2_mo + rdm2_mo.transpose(1, 0, 2, 3)) * 0.5

#    print("rdm1_mo_ful:",rdm1_mo)
#    print("rdm2_mo_ful:",rdm2_mo)

    n_vir = int(norbs-n_fro-nact)

    p = 0 
    q = 0
    for i in range(norbs):
        p += rdm1_mo[i,i]
        for j in range(norbs):
            q += rdm2_mo[i,i,j,j]
    print(p)
    print(q)
    penalty1 = p / nelec
    penalty2 = q / nelec/(nelec-1)
    print(penalty1)
    s = o = 0
    tei=numpy.transpose(tei,(0,2,3,1))
    term1 = np.einsum('pr,rq', oei, rdm1_mo)
    term11 = np.einsum('rp,rq', oei, rdm1_mo)
    rdm2_mo = numpy.transpose(rdm2_mo, (0, 2, 3, 1))
    term2 = np.einsum('ptrm,qtrm->pq', rdm2_mo, tei)
    term3 = np.einsum('tprm,tqrm->pq', rdm2_mo, tei)
    term4 = np.einsum('trpm,trqm->pq', rdm2_mo, tei)
    term5 = np.einsum('trmp,trmq->pq', rdm2_mo, tei)
    lagrangian = -0.5 * (term1.T + term11.T) - 0.25 * (term2 + term3 + term4 + term5)
   # tei=numpy.transpose(tei,(0,2,3,1))
    de = gradient_mo(mol, mo_coeff, oei, tei, rdm1_mo, rdm2_mo, mf, lagrangian, mymc)
    print(de)


