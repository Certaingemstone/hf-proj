# Restricted Hartree-Fock

import numpy as np
import psi4
from psi4.core import Molecule, BasisSet, Wavefunction, MintsHelper, Matrix, Vector

### Initialization ###

# define CO2
co2xyz = [[-1.1, 0, 0], [0, 0, 0], [1.1, 0, 0]]
co2elez = [8, 6, 8]
co2 = Molecule.from_arrays(geom=co2xyz, elez=co2elez)
N_elec = np.sum(co2elez)
mol = co2

assert N_elec % 2 == 0
assert mol.multiplicity() == 1
N_occ = N_elec // 2

energies = []

# Library calls to form the corresponding STO-3G basis
basisset = BasisSet.build(mol, target='sto-3g')

# and compute the nuclear repulsion energy from this geometry
ENuclear = mol.nuclear_repulsion_energy()

# and to compute on our basis set for the molecule...
mints = MintsHelper(basisset)
# one-electron Hamiltonian matrix elements
h = mints.ao_kinetic()
h.add(mints.ao_potential())
# overlap between spatial basis fns
S = mints.ao_overlap()
# electron repulsion integrals
ERI = mints.ao_eri().to_array() # nbf x nbf x nbf x nbf

wfn = Wavefunction.build(mol, basisset)

# use the core-Hamiltonian guess, where the initial C*C density matrix vanishes
Fock = h.clone()


### Orthogonalize the basis implicitly via canonical orthogonalization ###

dimS = S.shape[0]
# diagonalize overlap
S_eigvals = Vector(dimS)
U = Matrix(dimS,dimS)
S.diagonalize(U, S_eigvals, order = psi4.core.DiagonalizeOrder.Ascending)
Lambda = psi4.core.triplet(U, S, U, transA=True)

if np.any(np.less(np.diag(Lambda.to_array()), 1e-6)):
    print("WARNING: Small overlap eigenvalue. This code does not truncate orthogonalizing matrix. Be wary of numerical issues.")

# use the diagonal overlap matrix to compute S^-1/2 in diagonal basis
Lambda_invsqrt = Matrix.from_array(np.diag(np.power(np.diag(Lambda.to_array()), -0.5)))
# compute symmetric orthogonalizing matrix 
X = psi4.core.triplet(U, Lambda_invsqrt, U, transC=True)

### Diagonalize the Fock matrix in the orthogonal basis to obtain molecular orbitals, then return to our basis set ###

newE = 0
niter = 0

dimF = Fock.shape[0]
Fock_eigvals = Vector.from_array(np.zeros(dimF))
Fock_eigvects_Cprime = Matrix.from_array(np.zeros((dimF, dimF)))

while niter < 10:
    FockPrime = psi4.core.triplet(X, Fock, X, transA=True)
    FockPrime.diagonalize(Fock_eigvects_Cprime, Fock_eigvals, order = psi4.core.DiagonalizeOrder.Ascending)
    # return these molecular orbitals to our original basis
    C = psi4.core.doublet(X, Fock_eigvects_Cprime)

    ### Update the Fock matrix using the expansion coefficients in our basis set ###
    # Crop to only the occupied MOs
    C_np = C.to_array()
    C_occupied = C_np[:, 0:N_occ]
    # Compute density matrix
    Density = Matrix.from_array(np.matmul(C_occupied, C_occupied.T))
    # Compute the update to the non-core portion of the Hamiltonian
    # there's probably a way to use matrix multiplications?
    for mu in range(dimF):
        for nu in range(dimF):
            update = h.get(mu, nu)
            for rho in range(dimF):
                for sigm in range(dimF):
                    update += Density.get(rho, sigm) * (2 * ERI[mu][nu][rho][sigm] - ERI[mu][rho][nu][sigm])
            Fock.set(mu, nu, update)

    ### Calculate electronic energy ###
    newE = Density.vector_dot(h) + Density.vector_dot(Fock) + ENuclear
    energies.append(newE)
    niter += 1

print(energies)
# perform a check that the FC ~= SCe
SCe = psi4.core.triplet(S, C, Matrix.from_array(np.diag(Fock_eigvals.to_array())))

FC = psi4.core.doublet(Fock, C)

one = SCe.to_array()
two = FC.to_array()
print(two - 1)
