# CASCI-NNQS Forces Calculation

This repository contains the source code and example data used to evaluate
molecular nuclear forces from a CASCI-NNQS wave function. The code reconstructs
the one- and two-particle reduced density matrices (1-RDM and 2-RDM) from the
NNQS wave-function coefficients and combines them with molecular integrals and
orbital-response terms obtained with PySCF.

## Repository contents

```text
.
|-- test_forces.py          # Main program; submit/run this file for a complete calculation
|-- rdm.py                  # Construction and active-space expansion of the 1-RDM and 2-RDM
|-- gradient_integrals.py   # Derivatives of AO integrals and nuclear-repulsion energy
|-- rhf_mo_grad.py          # Molecular-orbital gradient/force evaluation and CPHF response
|-- pyscf_helper.py         # PySCF and OpenFermion integral/active-space utilities
|-- ci_psis/                # Example CASCI-NNQS wave-function files
`-- mo_coeff/               # Example PySCF molecular-orbital coefficient files
```

`test_forces.py` is the main entry point. The other four Python files are
supporting modules and must remain in the same directory when the calculation
is run.

## Requirements

- Python 3
- NumPy
- SciPy
- PySCF
- OpenFermion

For reproducible use, we recommend recording the exact Python and package
versions used for the published calculations in a pinned environment file or
`requirements.txt`.

## Input files

Each calculation requires two NumPy archives:

1. `ci_psis.npz`: CASCI-NNQS wavefunction data. The code reads the arrays
   `n_samples`, `samples`, and `psis` from this archive.
2. `mo_coeff.npz`: molecular-orbital coefficients generated with PySCF. The
   code reads the array `mo_coeff` from this archive.

Some example files are supplied.


The main program currently loads the literal filenames `ci_psis.npz` and
`mo_coeff.npz` from the working directory. Copy one matching pair to the
repository root and rename the copies before running the program. For example,
to run the LiF case in PowerShell:


The resulting RDM diagnostics and the final atom-by-Cartesian-component
force/gradient array are printed to standard output.

## Configuring a calculation

Before running `test_forces.py`, edit its main block so that all of the
following settings exactly match those used to generate the selected
`ci_psis.npz` and `mo_coeff.npz` files:

- atomic species and Cartesian geometry (`mol.atom`);
- basis set (`mol.basis`), total charge (`mol.charge`), and spin (`mol.spin`);
- frozen/core orbital indices (`fro_mo_list`);
- active orbital indices (`act_mo_list`);
- CASCI electron and orbital counts in `mcscf.CASCI`;
- `nact` and `n_qubit` (the latter is twice the number of spatial active
  orbitals in the current spin-orbital representation).

The orbital ordering in `ci_psis.npz` must be consistent with the
columns of `mo_coeff` and the chosen active space. A mismatch may produce a
numerical result without producing a clear runtime error.

## Typical workflow

1. Generate molecular-orbital coefficients with PySCF and save them as
   `mo_coeff.npz` with the key `mo_coeff`.
2. Generate or export the CASCI-NNQS wave function as `ci_psis.npz` with the
   keys `n_samples`, `samples`, and `psis`.
3. Set the molecular, basis-set, frozen-orbital, and active-space parameters in
   `test_forces.py` to match both input files.
4. Place the two input files in the repository root under the required literal
   filenames.
5. Run `python test_forces.py` (or submit that file through the Python execution
   command used by the computing cluster).

## Citation

If you use this code or the accompanying data, please cite the associated
paper. The full citation and DOI should be added here when they become
available.

## License

No reuse license is implied unless a `LICENSE` file is included. Before public
release, please add the license selected by the authors and confirm that any
adapted third-party code is compatible with that license and is appropriately
attributed.
