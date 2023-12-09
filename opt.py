from SCF import SCF

import numpy as np
from psi4.core import BasisSet

import csv

class LineSearch2D:
    # For performing and storing results of a naive line search through 2D configuration space over one dimension at a time. Should adapt easily to higher dimensions.
    def __init__(self, generator, N_elec: int, basis_target, xmin: float, xmax: float, ymin: float, ymax: float, Nx: int, Ny: int, SCF_convergence: float):
        self.xmin = xmin
        self.xmax = xmax
        self.last_x = None
        self.Nx = Nx
        self.ymin = ymin
        self.ymax = ymax
        self.last_y = None
        self.Ny = Ny
        self.N_elec = N_elec
        self.generator = generator # input (x, y) returns a Molecule
        self.basis = basis_target # string basis name
        self.convergence = SCF_convergence
        self.visited = [] # list of tuples of coordinates visited
        self.energies = []

    def get_best_visited(self, index: int):
        # yes, this is currently used inefficiently, should be optimized later
        amin = np.argmin(self.energies)
        print("Best energy:", self.energies[amin])
        return self.visited[amin][index]

    def change_roi(self, xmin, xmax, ymin, ymax, Nx=None, Ny=None, SCF_convergence=None):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.Nx = self.Nx if Nx is None else Nx
        self.Ny = self.Ny if Ny is None else Ny
        self.convergence = self.SCF_convergence if SCF_convergence is None else SCF_convergence

    def run_line(self, index: int, coordinate: float):
        """Performs a line search
        - index: The *FIXED* coordinate: 0 for x, 1 for y, 
        - coordinate: the value of x (if index 0) or y (if index 1) to run the search on"""
        if index == 1:
            self.last_y = coordinate
        else:
            self.last_x = coordinate
        var_arr = np.linspace(self.xmin, self.xmax, self.Nx) if index == 1 else np.linspace(self.ymin, self.ymax, self.Ny)
        for var_val in var_arr:
            mol = self.generator(var_val, coordinate) if index == 1 else self.generator(coordinate, var_val)
            basisset = BasisSet.build(mol, target=self.basis, quiet=True)
            HF = SCF(mol, self.N_elec, basisset)
            HF.iter_until(self.convergence)
            if index == 1:
                self.visited.append((var_val, coordinate))
            else:
                self.visited.append((coordinate, var_val))
            self.energies.append(HF.energies[-1])

    def search(self, x_convergence=0.001, y_convergence=0.001, max_iter=50):
        """Runs a line search alternating in x and y until a stable point or max_iter is reached.
        Then encloses the lowest energy point in a tighter ROI and repeats, until precision defined
        by x_ and y_convergence is reached."""
        x_diam = 2*(self.xmax - self.xmin) / self.Nx
        y_diam = 2*(self.ymax - self.ymin) / self.Ny
        print("Running initial fixed-x line search.")
        self.run_line(0, self.xmin) # run with x fixed
        y_fix = self.get_best_visited(1)
        print("Running initial fixed-y line search.")
        self.run_line(1, y_fix) # run with y fixed
        i = 0
        continuing = True
        changed_roi = False
        while continuing:

            # run line search with x fixed
            x_fix = self.get_best_visited(0)
            print("Searching at x:", x_fix)
            if abs(x_fix - self.last_x) < x_diam:
                # tighten ROI and continue
                print("Tightening ROI.")
                self.change_roi(x_fix - 4 * x_diam, x_fix + 4 * x_diam, y_fix - 4*y_diam, y_fix + 4*y_diam, SCF_convergence=max(self.convergence / 2, 1e-9))
                x_diam = 2*(self.xmax - self.xmin) / self.Nx
                y_diam = 2*(self.ymax - self.ymin) / self.Nx
                changed_roi = True
                # if we've reached precision demanded, we're done after the next search
                if x_diam < x_convergence and y_diam < y_convergence:
                    print("Optimization converged.")
                    continuing = False
            if changed_roi:
                self.run_line(0, self.xmin)
                changed_roi = False
            else:
                self.run_line(0, x_fix)

            # run line search with y fixed
            y_fix = self.get_best_visited(1)
            print("Searching at y:", y_fix)
            self.run_line(1, y_fix)
            i += 1
            if i > max_iter:
                print("WARNING: Line search did not converge to desired bounds.")
                break

    def save(self, filename, xname='length', yname='angle'):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow((xname, yname, 'energy'))
            for i, E in enumerate(self.energies):
                x, y = self.visited[i]
                writer.writerow((x, y, E))


