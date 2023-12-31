import hess
import numpy as np

MODE = None # SET THIS

co2_elez = [8, 6, 8]
h2o_elez = [1, 8, 1]

# equilibrium geometries from optimization
if MODE == "num":
    co2_coords = [[-1.18795, 0, 0], [0,0,0], [1.18795, 0, 0]]
    h2o_coords = [[0.98949, 0, 0], [0,0,0], [0.98949 * -0.173992, 0.98949 * 0.984747, 0]]

    print("Starting CO2 STO-3G")
    co2_sto3g_hess = hess.numhess(co2_coords, co2_elez, 'sto-3g', delta=0.001)
    np.save('co2hess_sto3g', co2_sto3g_hess)
    print("Starting H2O STO-3G")
    h2o_sto3g_hess = hess.numhess(h2o_coords, h2o_elez, 'sto-3g', delta=0.001)
    np.save('h2ohess_sto3g', h2o_sto3g_hess)

    
    co2_coords = [[-1.14328, 0, 0], [0,0,0], [1.14328, 0, 0]]
    h2o_coords = [[0.94725, 0, 0], [0,0,0], [0.94725 * -0.267407, 0.94725 * 0.963584, 0]]
    print("Starting CO2 6-31G*")
    co2 = hess.numhess(co2_coords, co2_elez, '6-31gs', delta=0.001)
    np.save('co2hess_6-31gs', co2)
    print("Starting H2O 6-31G*")
    h2o = hess.numhess(h2o_coords, h2o_elez, '6-31gs', delta=0.001)
    np.save('h2ohess_6-31gs', h2o)

# using the (semi-numerical) Hessian for HF that psi4 natively provides
elif MODE == "psi4":
    co2_coords = [[-1.14328, 0, 0], [0,0,0], [1.14328, 0, 0]]
    h2o_coords = [[0.94725, 0, 0], [0,0,0], [0.94725 * -0.267407, 0.94725 * 0.963584, 0]]
    co2hess_psi = hess.psi4hess(co2_coords, co2_elez, '6-31gs')
    hessian = co2hess_psi.to_array()
    np.save('co2hess_6-31gs_psi4', hessian)

    co2E, co2psi = hess.psi4freq(co2_coords, co2_elez, '6-31gs')
    print(co2E)
    print("VIBRATIONAL ANALYSIS")
    freqs = co2psi.frequencies()
    print(freqs.to_array())

    h2oE, h2opsi = hess.psi4freq(h2o_coords, h2o_elez, '6-31gs')
    print(h2oE)
    print("VIBRATIONAL ANALYSIS")
    freqs = h2opsi.frequencies()
    print(freqs.to_array())

# oops, it seems the Expanse allocation ran out
elif MODE == "dft":
    co2_coords = [[-1.160349, 0, 0], [0,0,0], [1.160349, 0, 0]]
    h2o_coords = [[0.96131, 0, 0], [0,0,0], [-0.24017, 0.93082, 0]]

    co2E, co2psi = hess.psi4freq_dft(co2_coords, co2_elez, 'cc-pVTZ', 'b3lyp')
    print(co2E)
    print("VIBRATIONAL ANALYSIS")
    print(co2psi.frequencies().to_array())

    h2oE, h2opsi = hess.psi4freq_dft(h2o_coords, h2o_elez, 'cc-pVTZ', 'b3lyp')
    print(h2oE)
    print("VIBRATIONAL ANALYSIS")
    print(h2opsi.frequencies().to_array())



