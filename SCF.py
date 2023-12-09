import numpy as np
import psi4
from psi4.core import MintsHelper, Matrix, Vector


class SCF:
    # For performing and retrieving information about HF SCF calculation
    def __init__(self, molecule: psi4.core.Molecule, N_elec: int, basis: psi4.core.BasisSet):
        # read molecule and basis details
        assert molecule.multiplicity() == 1
        self.mol = molecule
        assert N_elec % 2 == 0
        self.N_occ = N_elec // 2
        self.basis = basis
        # calculate integrals
        mints = MintsHelper(self.basis)
        self.ENuclear = self.mol.nuclear_repulsion_energy()
        h = mints.ao_kinetic()
        h.add(mints.ao_potential())
        self.h = h
        self.S = mints.ao_overlap() # n x n
        self.ERI = mints.ao_eri().to_array() # n x n x n x n two-electron integrals 
        self.nbf = self.S.shape[0]

        # prepare for iterations
        # use the core-Hamiltonian guess, where the initial C*C density matrix vanishes
        self.Fock = self.h.clone()
        self.Fock_eigvals = Vector.from_array(np.zeros(self.nbf))
        self.Fock_eigvects = Matrix.from_array(np.zeros((self.nbf, self.nbf)))

        # Orthogonalize the basis implicitly via canonical orthogonalization
        # diagonalize overlap
        _ = Vector(self.nbf)
        U = Matrix(self.nbf, self.nbf)
        self.S.diagonalize(U, _, order = psi4.core.DiagonalizeOrder.Ascending)
        Lambda = psi4.core.triplet(U, self.S, U, transA=True)
        if np.any(np.less(np.diag(Lambda.to_array()), 1e-6)):
            print("WARNING: Small overlap eigenvalue. This code does not truncate orthogonalizing matrix. Be wary of numerical issues.")
        # use the diagonal overlap matrix to compute S^-1/2 in diagonal basis
        Lambda_invsqrt = Matrix.from_array(np.diag(np.power(np.diag(Lambda.to_array()), -0.5)))
        # compute symmetric orthogonalizing matrix 
        self.X = psi4.core.triplet(U, Lambda_invsqrt, U, transC=True)

        self.C = None
        self.energies = []        


    def run_SCF_iteration(self):

        FockPrime = psi4.core.triplet(self.X, self.Fock, self.X, transA=True)
        FockPrime.diagonalize(self.Fock_eigvects, self.Fock_eigvals, order = psi4.core.DiagonalizeOrder.Ascending)
        # return these molecular orbitals to our original basis
        self.C = psi4.core.doublet(self.X, self.Fock_eigvects)

        ### Update the Fock matrix using the expansion coefficients in our basis set ###
        # Crop to only the occupied MOs
        C_np = self.C.to_array()
        C_occupied = C_np[:, 0:self.N_occ]
        # Compute density matrix
        Density = Matrix.from_array(np.matmul(C_occupied, C_occupied.T))
        # Compute the update to the non-core portion of the Hamiltonian
        # there's probably a way to use matrix multiplications?
        for mu in range(self.nbf):
            for nu in range(self.nbf):
                update = self.h.get(mu, nu)
                for rho in range(self.nbf):
                    for sigm in range(self.nbf):
                        update += Density.get(rho, sigm) * (2 * self.ERI[mu][nu][rho][sigm] - self.ERI[mu][rho][nu][sigm])
                self.Fock.set(mu, nu, update)

        ### Calculate electronic energy ###
        newE = Density.vector_dot(self.h) + Density.vector_dot(self.Fock) + self.ENuclear
        self.energies.append(newE)


    def iter_until(self, convergence_threshold: float, max_iter = 1000):
        self.run_SCF_iteration()
        self.run_SCF_iteration()
        for i in range(max_iter):
            self.run_SCF_iteration()
            if abs(self.energies[-1] - self.energies[-2]) < convergence_threshold:
                return
        print(f"WARNING: Convergence threshold {convergence_threshold} not reached after max_iter {max_iter} iterations.")
        print("Last energies:", self.energies[-10:-1])


    def check_consistency(self):
        SCe = psi4.core.triplet(self.S, self.C, Matrix.from_array(np.diag(self.Fock_eigvals.to_array())))
        FC = psi4.core.doublet(self.Fock, self.C)
        one = SCe.to_array()
        two = FC.to_array()
        print("SCe:", np.sum(abs(one)))
        print("FC:", np.sum(abs(two)))
        print("Difference:", np.sum(abs(two-one)))
