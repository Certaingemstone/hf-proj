from SCF import SCF

import psi4
from psi4.core import Molecule, BasisSet
import numpy as np


# Fully numerical Hessian at first-order finite difference
# Semi-numerical would require analytical gradients which I haven't implemented
# There exist fully analytical Hessians for RHF but I haven't done this

CONVERGENCE_HESS = 1e-12

def calc_energy(xyz, Z, basis):
    mol = Molecule.from_arrays(geom=xyz, elez=Z)
    basisset = BasisSet.build(mol, target=basis, quiet=True)
    HF = SCF(mol, sum(Z), basisset)
    HF.iter_until(CONVERGENCE_HESS)
    print("E:", HF.energies[-1])
    return HF.energies[-1]

def psi4hess(coords, elez, basis):
    """Use the analytical Hessian provided by psi4. Note unit is Ha/(Bohr^2)"""
    mol = psi4.core.Molecule.from_arrays(geom=coords, elez=elez)
    psi4.set_options({
        "basis": basis,
        "df_basis_scf": basis,
        "scf__reference": "rhf"
    })
    H = psi4.hessian('hf', molecule=mol, return_wft=False, irrep=-1)
    return H

def psi4freq(coords, elez, basis):
    mol = psi4.core.Molecule.from_arrays(geom=coords, elez=elez)
    psi4.set_options({
        "basis": basis,
        "df_basis_scf": basis,
        "scf__reference": "rhf"
    })
    E, wfn = psi4.frequency('hf', molecule=mol, return_wfn=True, irrep=-1)
    return E, wfn

def numhess(coords, elez, basis, delta=0.001):
    """For equilibrium coords and element Zs, compute numerical Hessian in basis target string"""
    
    div = 4 * (delta * delta) 
    ndof = 3 * len(elez)
    ret = np.zeros((ndof, ndof))

    # calculate equilibrium energy
    E0 = calc_energy(coords, elez, basis)

    # calculate diagonal entries
    for i in range(ndof):

        # initialize perturbed molecules and get energies
        pos = np.array(coords).flatten()
        pos[i] += 2 * delta
        print("POSITION", pos)
        Epos = calc_energy(pos, elez, basis)

        neg = np.array(coords).flatten()
        neg[i] -= 2 * delta
        print("POSITION", neg)
        Eneg = calc_energy(neg, elez, basis)
        
        # compute the second derivative
        ret[i][i] = (Epos - 2*E0 + Eneg) / div

        print(f"Computed diagonal {i+1} of {ndof}")
    
    # calculate upper triangular entries
    idx = 1
    tot = (ndof * ndof - ndof) // 2
    for i in range(ndof):
        for j in range(1 + i, ndof):
            # initialize perturbed molecules

            # energy calculations - TODO: parallel these
            pospos = np.array(coords).flatten()
            pospos[i] += delta
            pospos[j] += delta
            Epp = calc_energy(pospos, elez, basis)
            print("POSITIONpp", pospos)
            
            negpos = np.array(coords).flatten()
            negpos[i] -= delta
            negpos[j] += delta
            Enp = calc_energy(negpos, elez, basis)
            print("POSITIONnp", negpos)
            
            posneg = np.array(coords).flatten()
            posneg[i] += delta
            posneg[j] -= delta
            Epn = calc_energy(posneg, elez, basis)
            print("POSITIONpn", posneg)

            negneg = np.array(coords).flatten()
            negneg[i] -= delta
            negneg[j] -= delta
            Enn = calc_energy(negneg, elez, basis)
            print("POSITIONnn", negneg)
            
            Hij = (Epp - Enp - Epn + Enn) / div

            ret[i][j] = Hij
            ret[j][i] = Hij

            print(f"Computed off-diagonal {idx} of {tot}")
            idx += 1

    return ret

