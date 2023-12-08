# Restricted Hartree-Fock
from SCF import SCF

import numpy as np
from psi4.core import Molecule, BasisSet

import csv

CONVERGENCE = 2e-7 # Hartrees

# define CO2 line search
co2_bondlengths = [1.4 - i * 0.003 for i in range(100)]
co2xyz = [[[-length, 0, 0], [0, 0, 0], [length, 0, 0]] for length in co2_bondlengths]
co2elez = [8, 6, 8]
co2_opt_energies = []

for idx, xyz in enumerate(co2xyz):
    print("Computing energy for CO2 bond length co2_bondlength:", co2_bondlengths[idx])
    co2 = Molecule.from_arrays(geom=xyz, elez=co2elez)
    N_elec = np.sum(co2elez)
    basisset = BasisSet.build(co2, target='sto-3g', quiet=idx > 0)

    # set up the calculation
    co2_HF = SCF(co2, N_elec, basisset)
    co2_HF.run_SCF_iteration()
    co2_HF.run_SCF_iteration()

    while abs(co2_HF.energies[-1] - co2_HF.energies[-2]) > CONVERGENCE:
        co2_HF.run_SCF_iteration()

    co2_HF.check_consistency()
    print(co2_HF.energies)
    co2_opt_energies.append(co2_HF.energies[-1])

with open('co2line.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(("length", "energy"))
    for lengthenergy in zip(co2_bondlengths, co2_opt_energies):
        writer.writerow(lengthenergy)
