import tenpy
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain

#  1. define the model (transverse-field Ising) 
L = 30  # number of sites

model_params = {
    "L": L,
    "J": 1.0,        # interaction strength
    "g": 1.0,        # transverse field (critical point J = g)
    "bc_MPS": "finite",
}

M = TFIChain(model_params)

#  2. initial MPS: all spins up 
psi = MPS.from_lat_product_state(M.lat, [["up"]])

#  3. DMRG parameters 
dmrg_params = {
    "mixer": None,                # set to True later if it gets stuck
    "max_E_err": 1e-9,
    "max_sweeps": 6,
    "trunc_params": {
        "chi_max": 80,
        "svd_min": 1e-10,
    },
    "verbose": True,
}

#  4. run DMRG 
engine = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
E0, psi = engine.run()  # psi is optimized in-place

print("Ground-state energy:", E0)

#  5. look at entanglement (half-chain) 
S = psi.entanglement_entropy()      # entanglement for each bond
half_cut = (L - 1) // 2
print("Half-chain entanglement entropy S(L/2):", S[half_cut])

#  6. simple local observable: <σ^z_i> profile 
Z = psi.expectation_value("Sigmaz")
print("On-site<Z>:",Z)