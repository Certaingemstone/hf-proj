import opt

import numpy as np
from psi4.core import Molecule


water_elez = [1, 8, 1]
N_elec = 10
def water_generator(L, theta):
    O1 = [0,0,0]
    H1 = [L,0,0]
    H2 = [L*np.cos(theta), L*np.sin(theta), 0]
    coords = [H1, O1, H2]
    return Molecule.from_arrays(geom=coords, elez=water_elez)

Lmin = 0.8
Lmax = 1.2
NL = 40
tmin = 1.73
tmax = 1.96
Nt = 40

LS = opt.LineSearch2D(water_generator, N_elec, '6-31g', Lmin, Lmax, tmin, tmax, NL, Nt, 1e-6)
LS.search(max_iter=60)

LS.save("h2o_6-31g.csv")
