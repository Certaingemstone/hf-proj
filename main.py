# Restricted Hartree-Fock

import numpy as np
import psi4
from psi4.core import Molecule, BasisSet, Wavefunction, MintsHelper, Matrix, Vector

### Initialization ###

# define CO2
co2xyz = [[-1.2, 0, 0], [0, 0, 0], [1.2, 0, 0]]
co2elez = [8, 6, 8]
co2 = Molecule.from_arrays(geom=co2xyz, elez=co2elez)

mol = co2

assert mol.multiplicity() == 1

# Library calls to form the corresponding STO-3G basis
basisset = BasisSet.build(mol, target='sto-3g')

# and compute the nuclear repulsion energy from this geometry
# EN = mol.nuclear_repulsion_energy()

# and to compute on our basis set for the molecule...
mints = MintsHelper(basisset)
# one-electron Hamiltonian matrix elements
h = mints.ao_kinetic()
# h.add(mints.ao_potential())
# overlap between spatial basis fns
S = mints.ao_overlap()
# electron repulsion integrals
# ERI = mints.ao_eri()

wfn = Wavefunction.build(mol, basisset)

# use the core-Hamiltonian guess, where the initial C*C density matrix vanishes
Fock = h


### Orthogonalize the basis implicitly via canonical orthogonalization ###

dimS = S.shape[0]
# diagonalize overlap
S_eigvals = Vector.from_array(np.zeros(dimS))
S_eigvects = Matrix.from_array(np.zeros((dimS,dimS)))
S.diagonalize(S_eigvects, S_eigvals, order = psi4.core.DiagonalizeOrder.Ascending)
S_eigvects_inv = S_eigvects.clone()
S_eigvects_inv.invert()
S_diag = psi4.core.triplet(S_eigvects_inv, S, S_eigvects)
S_diag_elems = np.diag(S_diag.to_array())

if np.any(np.less(S_diag_elems, 1e-6)):
    print("WARNING: Small overlap eigenvalue. This code does not truncate orthogonalizing matrix. Be wary of numerical issues.")

# use the diagonal overlap matrix to compute S^-1/2 in diagonal basis
S_diag_invsqrt = Matrix.from_array(np.diag(np.power(S_diag_elems, -0.5)))
# compute canonical orthogonalizing matrix 
X = psi4.core.doublet(S_eigvects, S_diag_invsqrt)
X_inv = X.clone()
X_inv.invert()

### Diagonalize the Fock matrix in the orthogonal basis to obtain molecular orbitals, then return to our basis set ###

FockPrime = psi4.core.triplet(X_inv, Fock, X)
dimF = FockPrime.shape[0]
Fock_eigvals = Vector.from_array(np.zeros(dimF))
Fock_eigvects = Matrix.from_array(np.zeros((dimF, dimF)))
FockPrime.diagonalize(Fock_eigvects, Fock_eigvals, order = psi4.core.DiagonalizeOrder.Ascending)
# return these molecular orbitals to our original basis
C = psi4.core.doublet(X, Fock_eigvects)

### Update the Fock matrix using the expansion coefficients in our basis set ###


