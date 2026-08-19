import time
from functools import reduce
import numpy as np
from pyscf.fci.fci_slow import reorder_rdm
from pyscf.fci import cistring

def expand_strings(strs,n_orb):
    """
    For all cistrings in strs, get all possible cistrings wih one excitation.

    Args:
        n_orb (int): number of orbitals.
        strs (list): a list f given strings.

    Returns:
        _type_: _description_
    """
    orb_list = range(2*n_orb)
    def expand(str0):
        strs_list = [str0]
        occ = []
        vir = []
        for i in orb_list:
            if str0 & (1 << i):
                occ.append(i)
            else:
                vir.append(i)
        for i in occ:
            for a in vir:
                if i%2==a%2:
                    str1 = str0 ^ (1 << i) | (1 << a)
                    strs_list.append(str1)
        return strs_list
    strs_set = set()
    for s in strs.astype(np.int64):
        strs_list = expand(s)
        for string in strs_list:
            strs_set.add(string)
    strs_new = list(strs_set)
    strs_new.sort()
    strs_new = np.array(strs_new)
    return strs_new

def gen_linkstr_index_qubit(orb_list, strs_all,strs_simple):
    """
    Look up table, for the strings relationship in terms of a
    creation-annihilating operator pair.
    
    provide two string lists:
        strs_all is the strings expanded with the expand_strings function.
        strs is the original string list.

    Args:
        n_orb (int): number of orbitals.
        strs_all (list): a list of expanded strings.
        strs (list, optional): the original strings. Defaults to None.

    Returns:
        _type_: _description_
    """
    if strs_simple is None:
        strs_simple = strs_all
    strdic_all = dict(zip(strs_all,range(len(strs_all))))
    idx = []
    for str0 in strs_simple:
        idx.append(strdic_all[str0])
    strdic = dict(zip(strs_simple,idx))
    def propagate1e(str0):
        occ = []
        vir = []
        for i in orb_list:
            if str0 & (1 << i):
                occ.append(i)
            else:
                vir.append(i)
        linktab = []
        for i in occ:
            linktab.append((i, i, strdic_all[str0], 1))
        for i in occ:
            for a in vir:
                if i%2==a%2:
                    str1 = str0 ^ (1 << i) | (1 << a)
                    try:
                        addr = strdic[str1]
                        linktab.append((a, i, addr, \
                                    cistring.cre_des_sign(a, i, str0)))
                    except KeyError:
                        continue
                else:
                    pass
        return linktab

    t = [propagate1e(s) for s in strs_all.astype(np.int64)]
    return t


def make_rdm12(vec, strs, norb):
    """make rdm1 and rdm2"""
    strs_all = expand_strings(strs,n_orb=norb)
    strdic = dict(zip(strs_all,range(len(strs_all))))
    vec_all = np.zeros_like(strs_all,dtype="complex128")
    for i, coeff in enumerate(vec):
        str_tmp = strs[i]
        idx = strdic[str_tmp]
        vec_all[idx] = coeff
    print("start making link_index")
    time0 = time.perf_counter()
    link_index = gen_linkstr_index_qubit(range(2*norb),strs_all = strs_all,strs_simple = strs)
    time1 = time.perf_counter()
    print("end making link_index, time used: ", (time1 - time0))
    rdm1 = np.zeros((norb,norb),dtype = "complex128")
    rdm2 = np.zeros((norb,norb,norb,norb),dtype = "complex128")
    for str0, _ in enumerate(link_index):
        t1 = np.zeros((norb,norb),dtype = "complex128")
        for a, i, str1, sign in link_index[str0]:
            t1[i//2,a//2] += sign * vec_all[str1]
        rdm1+=vec_all[str0].conj()*t1
        rdm2 += np.einsum('ij,kl->jikl', t1.conj(), t1)
    rdm1 = rdm1.astype("float64")
    rdm2 = rdm2.astype("float64")
    return reorder_rdm(rdm1, rdm2)

def get_rdm1_from_casdm1(casdm1,ncore,nact,nmo):     # for MO cases
    rdm1 = np.zeros((nmo,nmo))
    for i in range(ncore):
        rdm1[i,i] = 2
    rdm1[ncore:ncore+nact,ncore:ncore+nact] = casdm1
    return rdm1

def get_rdm2_from_casdm(casdm1,casdm2,ncore,nact,nmo):    #for MO cases
    rdm2 = np.zeros((nmo,nmo,nmo,nmo))
    for i in range(ncore):
        for j in range(ncore):
            rdm2[i,i,j,j] += 4
            rdm2[i,j,j,i] -= 2
        rdm2[i,i,ncore:ncore+nact,ncore:ncore+nact] = casdm1 * 2
        rdm2[ncore:ncore+nact,ncore:ncore+nact,i,i] = casdm1 * 2
        rdm2[i,ncore:ncore+nact,ncore:ncore+nact,i] =-casdm1
        rdm2[ncore:ncore+nact,i,i,ncore:ncore+nact] =-casdm1
    rdm2[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact] = casdm2
    return rdm2

