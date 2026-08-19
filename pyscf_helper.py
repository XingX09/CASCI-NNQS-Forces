import functools

import numpy
import scipy.stats
import scipy.linalg
import pyscf
import pyscf.lo
import pyscf.symm
import pyscf.cc
import pyscf.fci
import openfermion

def _get_hamiltonian_ferm_op_from_mo_ints_mp_worker(args_worker):
    global two_body_mo_pqrs_global
    start_idx = args_worker[0]
    end_idx = args_worker[1]
    eps = args_worker[2]
    n_orb = two_body_mo_pqrs_global.shape[0]
    hamiltonian_ferm_op_2_worker = openfermion.FermionOperator()
    for p in range(start_idx, end_idx):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    if abs(two_body_mo_pqrs_global[p, q, r, s]) < eps:
                        continue
                    pa = p * 2
                    pb = p * 2 + 1
                    qa = q * 2
                    qb = q * 2 + 1
                    ra = r * 2
                    rb = r * 2 + 1
                    sa = s * 2
                    sb = s * 2 + 1
                    hamiltonian_ferm_op_2_worker += openfermion.FermionOperator(
                        ((pa, 1), (qa, 1), (ra, 0), (sa, 0)),
                        two_body_mo_pqrs_global[p][q][r][s] * 0.5
                    )
                    hamiltonian_ferm_op_2_worker += openfermion.FermionOperator(
                        ((pb, 1), (qb, 1), (rb, 0), (sb, 0)),
                        two_body_mo_pqrs_global[p][q][r][s] * 0.5
                    )
                    hamiltonian_ferm_op_2_worker += openfermion.FermionOperator(
                        ((pa, 1), (qb, 1), (rb, 0), (sa, 0)),
                        two_body_mo_pqrs_global[p][q][r][s] * 0.5
                    )
                    hamiltonian_ferm_op_2_worker += openfermion.FermionOperator(
                        ((pb, 1), (qa, 1), (ra, 0), (sb, 0)),
                        two_body_mo_pqrs_global[p][q][r][s] * 0.5
                    )
    return hamiltonian_ferm_op_2_worker

def get_hamiltonian_ferm_op_from_mo_ints(
        one_body_mo: numpy.ndarray,
        two_body_mo: numpy.ndarray,
        eps: float = 0.0):
    """
    Construct the one- and two-body terms of the Hamiltonian for a given
    one-electron MO integral in (p+, q) order and a give two-electron MO
    integral in (p+, s, q+, r) order (PySCF's ordering).

    Args:
        one_body_mo (numpy.ndarray): one-electron integral in (p+, q) order.
        two_body_mo (numpy.ndarray): two-electron integral in
            (p+, s, q+, r) order.
        eps (float): cut-off threshold.

    Notes:
        The integrals are for spatial-orbitals.
    """
    global two_body_mo_pqrs_global
    two_body_mo_pqrs = numpy.moveaxis(
        two_body_mo, [0, 2, 3, 1], [0, 1, 2, 3])
    hamiltonian_ferm_op_1 = openfermion.FermionOperator()
    hamiltonian_ferm_op_2 = openfermion.FermionOperator()
    for (p, q) in zip(*((abs(one_body_mo) > eps).nonzero())):
        p = int(p)
        q = int(q)
        pa = p * 2
        pb = p * 2 + 1
        qa = q * 2
        qb = q * 2 + 1
        hamiltonian_ferm_op_1 += openfermion.FermionOperator(
            ((pa, 1), (qa, 0)),
            one_body_mo[p][q]
        )
        hamiltonian_ferm_op_1 += openfermion.FermionOperator(
            ((pb, 1), (qb, 0)),
            one_body_mo[p][q]
        )
    for (p, q, r, s) in zip(*((abs(two_body_mo_pqrs) > eps).nonzero())):
        p = int(p)
        q = int(q)
        r = int(r)
        s = int(s)
        pa = p * 2
        pb = p * 2 + 1
        qa = q * 2
        qb = q * 2 + 1
        ra = r * 2
        rb = r * 2 + 1
        sa = s * 2
        sb = s * 2 + 1
        hamiltonian_ferm_op_2 += openfermion.FermionOperator(
            ((pa, 1), (qa, 1), (ra, 0), (sa, 0)),
            two_body_mo_pqrs[p][q][r][s] * 0.5
        )
        hamiltonian_ferm_op_2 += openfermion.FermionOperator(
            ((pb, 1), (qb, 1), (rb, 0), (sb, 0)),
            two_body_mo_pqrs[p][q][r][s] * 0.5
        )
        hamiltonian_ferm_op_2 += openfermion.FermionOperator(
            ((pa, 1), (qb, 1), (rb, 0), (sa, 0)),
            two_body_mo_pqrs[p][q][r][s] * 0.5
        )
        hamiltonian_ferm_op_2 += openfermion.FermionOperator(
            ((pb, 1), (qa, 1), (ra, 0), (sb, 0)),
            two_body_mo_pqrs[p][q][r][s] * 0.5
        )
    hamiltonian_ferm_op_1 = openfermion.normal_ordered(hamiltonian_ferm_op_1)
    hamiltonian_ferm_op_2 = openfermion.normal_ordered(hamiltonian_ferm_op_2)
    return hamiltonian_ferm_op_1, hamiltonian_ferm_op_2


def get_hamiltonian_ferm_op_from_mo_ints_mp(
        one_body_mo: numpy.ndarray,
        two_body_mo: numpy.ndarray,
        eps: float = 0.0,
        n_procs: int = 1):
    global two_body_mo_pqrs_global
    two_body_mo_pqrs_global = numpy.moveaxis(
        two_body_mo, [0, 2, 3, 1], [0, 1, 2, 3])
    hamiltonian_ferm_op_1 = openfermion.FermionOperator()
    hamiltonian_ferm_op_2 = openfermion.FermionOperator()
    for (p, q) in zip(*((abs(one_body_mo) > eps).nonzero())):
        p = int(p)
        q = int(q)
        pa = p * 2
        pb = p * 2 + 1
        qa = q * 2
        qb = q * 2 + 1
        hamiltonian_ferm_op_1 += openfermion.FermionOperator(
            ((pa, 1), (qa, 0)),
            one_body_mo[p][q]
        )
        hamiltonian_ferm_op_1 += openfermion.FermionOperator(
            ((pb, 1), (qb, 0)),
            one_body_mo[p][q]
        )
    n_orb = two_body_mo.shape[0]
    n_workers = min(n_procs, n_orb)
    if (n_workers != n_procs):
        print("Warning: change n_procs to %d" % (n_workers))

    chunk_size = n_orb // n_workers
    chunk_list = [chunk_size for i in range(n_workers)]
    for i in range(n_orb - chunk_size * n_workers):
        chunk_list[i] += 1

    import multiprocessing
    args_workers = []
    start_idx = 0
    end_idx = 0
    for i in range(n_workers):
        start_idx = end_idx
        end_idx += chunk_list[i]
        args_workers.append((start_idx, end_idx, eps))

    Pool = multiprocessing.Pool(n_workers)
    map_result = Pool.map(_get_hamiltonian_ferm_op_from_mo_ints_mp_worker,
                          args_workers)
    Pool.close()
    Pool.join()

    hamiltonian_ferm_op_2 = openfermion.FermionOperator()
    for i in range(n_workers):
        hamiltonian_ferm_op_2 = hamiltonian_ferm_op_2 + map_result[i]

    hamiltonian_ferm_op_1 = openfermion.normal_ordered(hamiltonian_ferm_op_1)
    hamiltonian_ferm_op_2 = openfermion.normal_ordered(hamiltonian_ferm_op_2)
    return hamiltonian_ferm_op_1, hamiltonian_ferm_op_2




def get_mo_integrals_from_molecule_and_hf_orb(     
        mol: pyscf.gto.Mole,
        mo_coeff: numpy.ndarray,
        debug: bool = False,
        hcore: numpy.ndarray = None):    
    """
    Notes:
        For two_body_mo, the current order is (ps|qr). A transpose like
        numpy.moveaxis(two_body_mo, [0, 2, 3, 1], [0, 1, 2, 3]) is necessary
        to get the (p+, q+, r, s) order or [p, q, r, s] indexing.
    """

    if hcore is None:
        hcore = mol.intor("int1e_nuc") + mol.intor("int1e_kin")  
    one_body_mo = functools.reduce(numpy.dot, (mo_coeff.T, hcore, mo_coeff))  
    eri = mol.intor("int2e") 
    two_body_mo = None
    try:
        import opt_einsum
        two_body_mo = opt_einsum.contract(
            "ijkl, ip, js, kq, lr->psqr",
            eri, mo_coeff.conj(), mo_coeff,
            mo_coeff.conj(), mo_coeff)    
            
    except ModuleNotFoundError:
        try:
            two_body_mo = numpy.einsum(
                "ijkl, ip, js, kq, lr->psqr",
                eri, mo_coeff.conj(), mo_coeff,
                mo_coeff.conj(), mo_coeff)
        except ValueError:
            print(
                "The system is too large. Please install opt_einsum and re-run init_scf().")
            raise ValueError

    if debug:
        two_body_mo_check = pyscf.ao2mo.restore(1, pyscf.ao2mo.get_mo_eri(
            mol, mo_coeff, compact=False),
            mol.nao_nr()
        )
        error = numpy.linalg.norm(two_body_mo - two_body_mo_check)
        assert(numpy.isclose(error, 0.0))
    return one_body_mo, two_body_mo


def get_spin_integrals_from_mo(one_body_mo: numpy.ndarray,
                               two_body_mo: numpy.ndarray):
    """
    Get the spin-orbital integrals from MO integrals.

    Notes:
        The output two_body_int is in (p+, q+, r, s) order.
    """
    n_orb = one_body_mo.shape[0]
    one_body_int = numpy.zeros([n_orb * 2] * 2)
    two_body_int = numpy.zeros([n_orb * 2] * 4)

    # # Original implementation.
    # for (p, q) in zip(*((abs(one_body_mo) > eps).nonzero())):
    #     one_body_int[2 * p][2 * q] = one_body_mo[p][q]
    #     one_body_int[2 * p + 1][2 * q + 1] = one_body_mo[p][q]
    # for (p, q, r, s) in zip(*((abs(two_body_mo) > eps).nonzero())):
    #     two_body_int[2 * p][2 * q][2 * r][2 * s] = \
    #         two_body_mo[p][s][q][r]
    #     two_body_int[2 * p + 1][2 * q + 1][2 * r + 1][2 * s + 1] = \
    #         two_body_mo[p][s][q][r]
    #     two_body_int[2 * p + 1][2 * q][2 * r][2 * s + 1] = \
    #         two_body_mo[p][s][q][r]
    #     two_body_int[2 * p][2 * q + 1][2 * r + 1][2 * s] = \
    #         two_body_mo[p][s][q][r]

    # Taking advantage of numpy's vectorization.
    one_body_int[0::2, 0::2] = one_body_mo
    one_body_int[1::2, 1::2] = one_body_mo
    two_body_mo_pqrs = numpy.moveaxis(two_body_mo, [0, 2, 3, 1], [0, 1, 2, 3])
    two_body_int[0::2, 0::2, 0::2, 0::2] = two_body_mo_pqrs
    two_body_int[1::2, 1::2, 1::2, 1::2] = two_body_mo_pqrs
    two_body_int[1::2, 0::2, 0::2, 1::2] = two_body_mo_pqrs
    two_body_int[0::2, 1::2, 1::2, 0::2] = two_body_mo_pqrs
    return one_body_int, two_body_int


def get_localized_mo_coeff(mf: pyscf.scf.rhf.RHF, molecule: pyscf.gto.Mole,
                           localized_orbitals: str = "None"):

    if localized_orbitals == "None":
        localized_orbitals = None

    mo_coeff = mf.mo_coeff.copy()
    basis = molecule.basis
    n_mo_occ = numpy.count_nonzero(mf.mo_occ > 0)
    if localized_orbitals is not None:
        if localized_orbitals in ["iao", "IAO"]:
            print("Use IAO localization.")

            if basis != "sto3g":
                print("%s at large basis may lead to incorrect result. \
This may be fixed in the future." % (localized_orbitals))

            mo_occ = mf.mo_coeff[:, :]  # mf.mo_coeff[:, :n_mo_occ]
            a = pyscf.lo.iao.iao(molecule, mo_occ)
            a = pyscf.lo.vec_lowdin(a, mf.get_ovlp())
            mo_occ = a.T.dot(mf.get_ovlp().dot(mo_occ))
            mo_coeff = a.copy()

        elif localized_orbitals in ["ibo", "IBO"]:
            print("Use IBO localization.")

            if basis != "sto3g":
                print("%s at large basis may lead to incorrect result. \
This may be fixed in the future." % (localized_orbitals))

            mo_occ = mf.mo_coeff[:, :]
            a = pyscf.lo.iao.iao(molecule, mo_occ)
            a = pyscf.lo.vec_lowdin(a, mf.get_ovlp())
            a = pyscf.lo.ibo.ibo(molecule, mo_occ, iaos=a, s=mf.get_ovlp())
            a = pyscf.lo.vec_lowdin(a, mf.get_ovlp())
            mo_occ = a.T.dot(mf.get_ovlp().dot(mo_occ))
            mo_coeff = a.copy()
        elif localized_orbitals in ["cholesky", "Cholesky", "Cho"]:
            print("Using cholesky localization.")
            mo_coeff_occupied = pyscf.lo.cholesky_mos(
                mf.mo_coeff[:, :n_mo_occ])
            mo_coeff_virtual = pyscf.lo.cholesky_mos(
                mf.mo_coeff[:, n_mo_occ:])
            mo_coeff = numpy.hstack((mo_coeff_occupied, mo_coeff_virtual))
        elif localized_orbitals in ["NAO", "nao"]:
            print("Using NAO localization.")
            mo_coeff = pyscf.lo.orth_ao(mf, method="nao")
        elif localized_orbitals in ["er", "ER", "Edmiston-Ruedenberg"]:
            print("Using Edmiston-Ruedenberg localization.")
            mo_coeff = pyscf.lo.ER(molecule, mo_coeff=mf.mo_coeff).kernel()
            mo_coeff = pyscf.lo.vec_lowdin(mo_coeff, mf.get_ovlp())
        elif localized_orbitals in ["boys", "Boys", "Foster-Boys"]:
            print("Using Foster-Boys localization.")
            mo_coeff = pyscf.lo.Boys(molecule, mo_coeff=mf.mo_coeff).kernel()
            mo_coeff = pyscf.lo.vec_lowdin(mo_coeff, mf.get_ovlp())
        elif type(localized_orbitals) is numpy.ndarray:
            assert(localized_orbitals.shape == mo_coeff.shape)
            mo_coeff = mo_coeff.dot(localized_orbitals)
        else:
            raise ValueError("Localization orbital %s \
not supported!" % (localized_orbitals))
    return mo_coeff


def get_active_space_effective_mo_integrals(
        one_body_mo: numpy.ndarray,
        two_body_mo: numpy.ndarray,
        freeze_indices_mo: list = None,
        active_indices_mo: list = None):
    """
    Calculate active-space integrals with PySCF ordering.

    Notes:
        The indices and integrals are for spatial MOs.

    References:
        [1]. Emiel Koridon et al. PHYSICAL REVIEW RESEARCH 3(2021), 033127
    """
    one_body_mo_active = None
    two_body_mo_active = None
    core_correction = 0.0
    if freeze_indices_mo is None:
        freeze_indices_mo = []
    if active_indices_mo is None:
        one_body_mo_active = one_body_mo.copy()
        two_body_mo_active = two_body_mo.copy()
    else:
        one_body_mo_new = numpy.copy(one_body_mo)
        for p in freeze_indices_mo:
            core_correction += 2. * one_body_mo[p][p]
            for q in freeze_indices_mo:
                core_correction += (2. * two_body_mo[q][q][p][p] -
                                    two_body_mo[q][p][p][q])
        for uu in active_indices_mo:
            for vv in active_indices_mo:
                for ii in freeze_indices_mo:
                    one_body_mo_new[uu][vv] += (
                        2. * two_body_mo[ii][ii][uu][vv] -
                        two_body_mo[ii][vv][uu][ii]
                    )
        one_body_mo_active = one_body_mo_new[numpy.ix_(
            active_indices_mo, active_indices_mo)]
        two_body_mo_active = two_body_mo[numpy.ix_(
            active_indices_mo, active_indices_mo,
            active_indices_mo, active_indices_mo)]
    return one_body_mo_active, two_body_mo_active, core_correction


def init_scf(geometry, basis="sto-3g", spin=0,
             freeze_indices_spatial=[],
             active_indices_spatial=[],
             run_fci: bool = False,
             localized_orbitals: str = None,
             use_symmetry: bool = False,
             override_symmetry_group: str = None,
             fermion_to_qubit_mapping: str = "jw",
             sort_orbital_for_dmrg: bool = False,
             run_rccsd: bool = False,
             return_spin_orb_int: bool = False,
             n_procs: int = 1,
             freeze_occ_indices_spin: list = None,
             freeze_vir_indices_spin: list = None):
    """
    Generate the system Hamiltonian and other quantities for a give molecule.

    Args:
        geometry (list): The structure of the molecule.
        basis (str): Basis set for SCF calculations.
        spin (int): Describes multiplicity of the molecular system.
        freeze_indices_spatial (list): Occupied indices (frozen orbitals)
            of spatial orbitals.
        active_indices_spatial (list): Active indices of spatial
            orbitals.
        run_fci (bool): Whether FCI calculation is performed.
        localized_orbitals (str): Whether to use localized orbitals. If
            is None, no localization if performed.
        use_symmetry (bool): Whether to use symmetry and return the character
            table of orbitals. Exclusive with localized_orbitals.
        override_symmetry_group (str): Override the symmetry point group
            determined by PySCF.
        fermion_to_qubit_mapping (str): The fermion-to-qubit mapping
            for Hamiltonian.
        sort_orbital_for_dmrg (bool): Reorder the orbitals according to its
            electronic state symmetry. Can be used together with the
            override_symmetry_group.
        run_rccsd (bool): Whether the RCCSD is performed.
        return_spin_orb_int (bool): Whether to return the one- and two-electron
            integrals in spin-orbital basis.
        n_procs (int): Number of processes to contruct the Hamiltonian.
        freeze_occ_indices_spin (list): Indices of spin orbitals which are
            assumed to be occupied.
        freeze_vir_indices_spin (list): Indices of spin orbitals which are
            assumed to be unoccupied.

    Returns:
        molecule (pyscf.gto.M object): Contains various properties
            of the system.
        n_qubits (int): Number of qubits in the Hamiltonian.
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        occ_indices_spin (int): Occupied indices of spin orbitals.
        hamiltonian_fermOp (openfermion.FermionOperator): Fermionic
            Hamiltonian.
        hamiltonian_qubitOp (openfermion.QubitOperator): Qubit Hamiltonian
            under JW transformation.
        orbsym (numpy.ndarray): The irreducible representation of each
            spatial orbital. Only returns when use_symmetry is True.
        prod_table (numpy.ndarray): The direct production table of orbsym.
            Only returns when use_symmetry is True.

    """
    eps = 0.0

    if localized_orbitals == "None":
        localized_orbitals = None

    if localized_orbitals is not None:
        if use_symmetry is True:
            print("Using localized orbitals will cause the returned orbsym \
and prod_table invalid. Handle with care!")

    molecule = pyscf.gto.M(
        atom=geometry,
        basis=basis,
        spin=spin
    )

    if use_symmetry or sort_orbital_for_dmrg:
        if override_symmetry_group is not None:
            molecule = pyscf.gto.M(
                atom=geometry,
                basis=basis,
                spin=spin,
                symmetry=override_symmetry_group
            )
        else:
            molecule = pyscf.gto.M(
                atom=geometry,
                basis=basis,
                spin=spin,
                symmetry=True
            )
        print("Use symmetry. Molecule point group: %s" % (molecule.topgroup))

    mf = pyscf.scf.RHF(molecule)
    print("Running RHF...")
    mf.kernel()
    mo_coeff = mf.mo_coeff

    mo_coeff = get_localized_mo_coeff(
        mf, molecule, localized_orbitals)

    # ***DEBUG***
    # u = _optimize_mo_coeff(mo_coeff)
    # print("mo_coeff Before: \n", mo_coeff)
    # mo_coeff = mo_coeff.dot(u)
    # print("mo_coeff After: \n", mo_coeff)

    if run_rccsd:
        print("Running RCCSD")
        mf_cc = pyscf.cc.RCCSD(mf)
        mf_cc.kernel()

    energy_RHF = mf.e_tot
    energy_nuc = molecule.energy_nuc()
    print("Hartree-Fock energy: %20.16f Ha" % (energy_RHF))
    if run_rccsd:
        energy_RCCSD = mf_cc.e_tot
        print("CCSD energy: %20.16f Ha" % (energy_RCCSD))

    if run_fci:
        mf_fci = pyscf.fci.FCI(mf)
        energy_fci = mf_fci.kernel()[0]
        print("FCI energy: %20.16f Ha" % (energy_fci))

    # return None # TODO WYJ
    n_orb = mo_coeff.shape[1]  # molecule.nao_nr()
    n_orb_occ = sum(molecule.nelec) // 2
    occ_indices_spin = [i for i in range(molecule.nelectron)]
    hcore = mf.get_hcore()
    one_body_mo, two_body_mo = get_mo_integrals_from_molecule_and_hf_orb(
        molecule, mo_coeff)
    fre_mo_list=[0,1]
    act_mo_list=[2,3,4,5,6,7,8,9]
    one_body_mo_active, two_body_mo_active, core_correction=get_active_space_effective_mo_integrals(one_body_mo,two_body_mo,freeze_indices_mo=fre_mo_list,active_indices_mo=act_mo_list)

    if len(freeze_indices_spatial) != 0 or len(active_indices_spatial) != 0:
        print("The freeze_indices_spatial and active_indices_spatial are\
currently not supported. Please use freeze_occ_indices_spin and\
freeze_vir_indices_spin instead.")


    hamiltonian_ferm_op_1, hamiltonian_ferm_op_2 = \
        get_hamiltonian_ferm_op_from_mo_ints_mp(
         #   one_body_mo, two_body_mo, eps,
            one_body_mo_active, two_body_mo_active, eps,
            n_procs=n_procs)

    hamiltonian_ferm_op = hamiltonian_ferm_op_1 + hamiltonian_ferm_op_2
    hamiltonian_ferm_op += energy_nuc + core_correction

    if sort_orbital_for_dmrg:
        print("Reordering orbitals according to orbital symmetry.")
        orbsym = mf.orbsym if hasattr(mf, "orbsym") else \
            numpy.zeros(n_orb, numpy.int32)
        sorted_args = list(numpy.argsort(orbsym))

        def _order_function(idx_ori: int, *args):
            idx_new = int(sorted_args.index(idx_ori // 2) * 2 + idx_ori % 2)
            return idx_new
        hamiltonian_ferm_op = openfermion.reorder(
            hamiltonian_ferm_op, order_function=_order_function)

        mo_occ_reordered = mf.mo_occ[sorted_args]
        occ_indices_spin = []
        for i in range(n_orb):
            if mo_occ_reordered[i] == 2:
                occ_indices_spin.append(2 * i)
                occ_indices_spin.append(2 * i + 1)
            elif mo_occ_reordered[i] == 1:
                occ_indices_spin.append(2 * i)
            else:
                pass

    if freeze_occ_indices_spin is not None and \
            freeze_vir_indices_spin is not None:
        hamiltonian_ferm_op = openfermion.freeze_orbitals(
            hamiltonian_ferm_op,
            occupied=freeze_occ_indices_spin,
            unoccupied=freeze_vir_indices_spin)
        n_qubits = openfermion.count_qubits(hamiltonian_ferm_op)
        occ_indices_spin_tmp = [
            i for i in occ_indices_spin
            if i not in freeze_occ_indices_spin + freeze_vir_indices_spin]
        min_occ_index_spin = min(occ_indices_spin_tmp)
        occ_indices_spin = [i - min_occ_index_spin
                            for i in occ_indices_spin_tmp]
        n_orb = n_qubits // 2
        n_orb_occ = len(occ_indices_spin) // 2 + len(occ_indices_spin) % 2-len(fre_mo_list)

    hamiltonian_qubit_op = None
    if fermion_to_qubit_mapping is not None:
        if fermion_to_qubit_mapping == "jw":
            hamiltonian_qubit_op = openfermion.jordan_wigner(hamiltonian_ferm_op)
        else:
            raise NotImplementedError("Fermion-to-qubit mapping {} not \
implemented.".format(fermion_to_qubit_mapping))
    n_qubits = n_orb * 2  # openfermion.count_qubits(hamiltonian_fermOp)

    returned_vals = [molecule, n_qubits, n_orb, n_orb_occ, occ_indices_spin,
                     hamiltonian_ferm_op, hamiltonian_qubit_op, one_body_mo, two_body_mo, mo_coeff]

    if use_symmetry:
        orbsym = mf.orbsym if hasattr(mf, "orbsym") else \
            numpy.zeros(n_orb, numpy.int32)
        n_sym_ops = numpy.max(orbsym)
        prod_table = pyscf.symm.direct_prod(
            numpy.arange(n_sym_ops + 1),
            numpy.arange(n_sym_ops + 1),
            molecule.topgroup if override_symmetry_group is None
            else override_symmetry_group)
        if freeze_occ_indices_spin is not None and \
                freeze_vir_indices_spin is not None:
            n_orb_ori = len(orbsym)
            except_indices_spin = freeze_occ_indices_spin + \
                freeze_vir_indices_spin
            orbsym = [orbsym[i] for i in range(n_orb_ori) if
                      (2 * i not in except_indices_spin and
                       2 * i + 1 not in except_indices_spin)]
            pass
        returned_vals.append(orbsym)
        returned_vals.append(prod_table)

    if return_spin_orb_int:
        one_body_int, two_body_int = get_spin_integrals_from_mo(
            one_body_mo=one_body_mo,
            two_body_mo=two_body_mo)
        returned_vals.append(one_body_int)
        returned_vals.append(two_body_int)

    # TODO: Change the first return value to openfermion's MolecularData
    return tuple(returned_vals)
