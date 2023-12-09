from SCF import SCF

from psi4.core import Molecule, BasisSet
import numpy as np

from multiprocessing import Pool

# Fully numerical Hessian at first-order finite difference
# Semi-numerical would require analytical gradients which I haven't implemented
# There exist fully analytical Hessians for RHF but I haven't done this

CONVERGENCE_HESS = 1e-9

def calc_energy(xyz, Z, basis):
    mol = Molecule.from_arrays(xyz, Z)
    basisset = BasisSet.build(mol, target=basis, quiet=True)
    HF = SCF(mol, sum(Z), basisset)
    HF.iter_until(CONVERGENCE_HESS)
    return HF.energies[-1]

def numhess(coords, elez, basis, delta=0.001):
    """For equilibrium coords and element Zs, compute numerical Hessian in basis target string"""
    
    div = 4 * delta ** 2 
    ndof = 3 * len(elez)
    ret = np.zeros((ndof, ndof))

    # calculate equilibrium energy
    E0 = calc_energy(coords, elez, basis)

    # calculate diagonal entries
    for i in range(ndof):

        # initialize perturbed molecules and get energies
        major = i // 3
        minor = i % 3

        pos = coords.copy()
        pos[major][minor] += 2 * delta
        Epos = calc_energy(pos, elez, basis)

        neg = coords.copy()
        neg[major][minor] -= 2 * delta
        Eneg = calc_energy(neg, elez, basis)
        
        # compute the second derivative
        ret[i][i] = (Epos - 2*E0 + Eneg) / div
    
    # calculate upper triangular entries
    for i in range(ndof):
        for j in range(1 + i, ndof):
            # initialize perturbed molecules
            majori = i // 3
            minori = i % 3
            majorj = j // 3
            minorj = j % 3

            needed = []

            pospos = coords.copy()
            pospos[majori][minori] += delta
            pospos[majorj][minorj] += delta
            needed.append((pospos, elez, basis))
            
            negpos = coords.copy()
            negpos[majori][minori] -= delta
            negpos[majorj][minorj] += delta
            needed.append((negpos, elez, basis))
            
            posneg = coords.copy()
            posneg[majori][minori] += delta
            posneg[majorj][minorj] -= delta
            needed.append((posneg, elez, basis))

            negneg = coords.copy()
            negneg[majori][minori] -= delta
            negneg[majorj][minorj] -= delta
            needed.append((negneg, elez, basis))
            
            E_pert = None
            with Pool(processes=4) as pool:
                E_pert = pool.starmap(calc_energy, needed)
            
            Hij = (E_pert[0] - E_pert[1] - E_pert[2] + E_pert[3]) / div

            ret[i][j] = Hij
            ret[j][i] = Hij
    return ret

