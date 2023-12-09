# Restricted Hartree-Fock
from SCF import SCF

import numpy as np
from psi4.core import Molecule, BasisSet

import csv
import os.path

CONVERGENCE = 2e-7 # Hartrees
CONVERGENCE_2 = 1e-9
BASIS = '6-31gs'

co2elez = [8, 6, 8]

if not os.path.isfile(f'co2line_coarse{BASIS}.csv'):
    # define CO2 line search
    co2_bondlengths = [1.4 - i * 0.003 for i in range(100)]
    co2xyz = [[[-length, 0, 0], [0, 0, 0], [length, 0, 0]] for length in co2_bondlengths]
    
    co2_opt_energies = []

    for idx, xyz in enumerate(co2xyz):
        print("Computing energy for CO2 bond length co2_bondlength:", co2_bondlengths[idx])
        co2 = Molecule.from_arrays(geom=xyz, elez=co2elez)
        N_elec = np.sum(co2elez)
        basisset = BasisSet.build(co2, target=BASIS, quiet=idx > 0)

        # set up the calculation
        co2_HF = SCF(co2, N_elec, basisset)
        co2_HF.iter_until(CONVERGENCE)
        # co2_HF.check_consistency()
        # print(co2_HF.energies)
        co2_opt_energies.append(co2_HF.energies[-1])

    with open(f'co2line_coarse{BASIS}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(("length", "energy"))
        for lengthenergy in zip(co2_bondlengths, co2_opt_energies):
            writer.writerow(lengthenergy)

# perform fine CO2 line search
if os.path.isfile(f'co2line_coarse{BASIS}.csv') and not os.path.isfile(f'co2line_fine{BASIS}.csv'):
    # define new search based on the coarse output
    co2_opt_energies = []
    co2_bondlengths = []
    with open(f'co2line_coarse{BASIS}.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader) # skip the header
        for row in reader:
            co2_bondlengths.append(float(row[0]))
            co2_opt_energies.append(float(row[1]))
    opt_idx = np.argmin(co2_opt_energies)
    center = co2_bondlengths[opt_idx]
    print("\nPERFORMING FINE LINE SEARCH\n")

    fine_bondlengths = np.linspace(center - 0.006, center + 0.006, num=40)
    fine_co2xyz = [[[-length, 0, 0], [0, 0, 0], [length, 0, 0]] for length in fine_bondlengths]
    fine_co2_opt_energies = []

    for xyz in fine_co2xyz:
        print("Computing energy for CO2 bond length:", xyz[-1][0])
        co2 = Molecule.from_arrays(geom=xyz, elez=co2elez)
        N_elec = np.sum(co2elez)
        basisset = BasisSet.build(co2, target='sto-3g', quiet=True)

        # set up the calculation
        co2_HF2 = SCF(co2, N_elec, basisset)
        co2_HF2.iter_until(CONVERGENCE_2)
        fine_co2_opt_energies.append(co2_HF2.energies[-1])

    with open(f'co2line_fine{BASIS}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(("length", "energy"))
        for lengthenergy in zip(fine_bondlengths, fine_co2_opt_energies):
            writer.writerow(lengthenergy)



