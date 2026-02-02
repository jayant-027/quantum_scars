import numpy as np
import h5py
from tenpy.models.pxp import PXPChain
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

# ---------------------------------------------------
# PARAMETERS
# ---------------------------------------------------
N = 45
k = 9
sweeps = 10
chi_max = 200
output_file = "dataset_N45.h5"

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------
model = PXPChain({"L": N, "conserve": None})
sites = model.lat.mps_sites()

# ---------------------------------------------------
# FEATURE EXTRACTION
# (no Hamiltonian used at all)
# ---------------------------------------------------
def extract_features(psi):
    # Magnetization: returns array in older TenPy
    mz_vals = psi.expectation_value("Sigmaz")
    mz = float(np.mean(mz_vals))   # take average magnetization

    # Entanglement entropy
    S = psi.entanglement_entropy()
    mid = len(S) // 2
    S_half = float(S[mid])
    S_avg = float(sum(S) / len(S))
    S_max = float(max(S))

    chi_state = max(psi.chi)
    label = 0

    return mz, S, S_half, S_avg, S_max, chi_state, label


# ---------------------------------------------------
# GROUND STATE
# ---------------------------------------------------
dmrg_options = {
    "max_sweeps": sweeps,
    "mixer": True,
    "chi_list": {0: chi_max},
    "trunc_params": {"svd_min": 1e-10},
}

print("Running ground-state DMRG...")

psi = MPS.from_product_state(sites, ["down"] * N, bc="finite")
info0 = dmrg.run(psi, model, dmrg_options)

ground_state = psi.copy()
ground_energy = float(info0["E"])

states = [(ground_state, ground_energy)]

# ---------------------------------------------------
# EXCITED STATES
# ---------------------------------------------------
print(f"Computing {k} excited states...")

for i in range(k):
    print(f"  -> Excited state #{i+1}")

    psi = MPS.from_product_state(sites, ["down"] * N, bc="finite")
    info_exc = dmrg.run(psi, model, dmrg_options)

    excited_energy = float(info_exc["E"])
    states.append((psi.copy(), excited_energy))

# ---------------------------------------------------
# SAVE DATASET
# ---------------------------------------------------
print("Saving dataset to", output_file, "...")

with h5py.File(output_file, "w") as f:
    for idx, (psi, energy) in enumerate(states):
        grp = f.create_group(f"state_{idx}")

        mz, S, S_half, S_avg, S_max, chi_state, label = extract_features(psi)

        grp.create_dataset("energy", data=energy)
        grp.create_dataset("magnetization", data=mz)
        grp.create_dataset("entropy", data=S)
        grp.create_dataset("entropy_half", data=S_half)
        grp.create_dataset("entropy_avg", data=S_avg)
        grp.create_dataset("entropy_max", data=S_max)
        grp.create_dataset("chi", data=chi_state)
        grp.create_dataset("label", data=label)

print("Dataset saved successfully!")
