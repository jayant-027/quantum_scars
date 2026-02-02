import numpy as np
import h5py
import scipy.sparse as sp

##############################################################################
# 1. Load ED eigenstates
##############################################################################

input_file = "pxp_L25_ed_states.h5"
output_file = "pxp_L25_krylov_scar_data.h5"

with h5py.File(input_file, "r") as f:
    eigenvalues = f["eigenvalues"][:]
    eigenvectors = f["eigenvectors"][:]
    L = int(f.attrs["L"])
    dim = int(f.attrs["basis_size"])

print("Loaded ED data")
print("L =", L)
print("Hilbert space dimension =", dim)

##############################################################################
# 2. Rebuild constrained basis
##############################################################################

def allowed(state):
    return (state & (state << 1)) == 0

basis = np.array([s for s in range(1 << L) if allowed(s)])
state_to_index = {s: i for i, s in enumerate(basis)}

assert len(basis) == dim
print("Constrained basis rebuilt")

##############################################################################
# 3. Build exact PXP Hamiltonian (sparse)
##############################################################################

rows, cols, data = [], [], []

for i, s in enumerate(basis):
    for site in range(L):
        if ((s >> site) & 1) == 0:
            if site > 0 and ((s >> (site - 1)) & 1):
                continue
            if site < L - 1 and ((s >> (site + 1)) & 1):
                continue

            flipped = s ^ (1 << site)
            j = state_to_index[flipped]

            rows.append(i)
            cols.append(j)
            data.append(1.0)

H = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))
print("PXP Hamiltonian built")

##############################################################################
# 4. Initial Néel state |1010...>
##############################################################################

neel_state = sum(1 << i for i in range(0, L, 2))
psi0 = np.zeros(dim)
psi0[state_to_index[neel_state]] = 1.0

##############################################################################
# 5. Build Krylov subspace of H
##############################################################################

max_krylov_dim = 20   # enough for scars
krylov_states = []

psi = psi0.copy()

for n in range(max_krylov_dim):
    norm = np.linalg.norm(psi)
    if norm < 1e-12:
        break

    psi = psi / norm

    # Orthogonalize
    for prev in krylov_states:
        psi -= np.dot(prev, psi) * prev

    norm = np.linalg.norm(psi)
    if norm < 1e-12:
        break

    psi = psi / norm
    krylov_states.append(psi.copy())

    psi = H @ psi

krylov_states = np.array(krylov_states)
n_krylov = krylov_states.shape[0]

print("Krylov subspace built")
print("Krylov dimension =", n_krylov)

##############################################################################
# 6. Compute overlaps with eigenstates
##############################################################################

n_eigs = eigenvectors.shape[1]
overlaps = np.zeros((n_eigs, n_krylov))

for i in range(n_eigs):
    psi_e = eigenvectors[:, i]
    for k in range(n_krylov):
        overlaps[i, k] = abs(np.dot(krylov_states[k], psi_e))**2

##############################################################################
# 7. Scar score = total Krylov weight
##############################################################################

scar_score = overlaps.sum(axis=1)

##############################################################################
# 8. Save results
##############################################################################

with h5py.File(output_file, "w") as f:
    f.create_dataset("eigenvalues", data=eigenvalues)
    f.create_dataset("scar_score", data=scar_score)
    f.create_dataset("overlaps", data=overlaps)
    f.attrs["L"] = L
    f.attrs["dim"] = dim
    f.attrs["krylov_dim"] = n_krylov
    f.attrs["description"] = "Scar score via Krylov subspace from Neel state"

print("Saved scar data to:", output_file)

##############################################################################
# 9. Print top scar candidates
##############################################################################

print("\nTop scar candidates:")
idx = np.argsort(-scar_score)
for i in idx[:6]:
    print(
        f"state {i:3d} | "
        f"E = {eigenvalues[i]: .6f} | "
        f"scar_score = {scar_score[i]:.4f}"
    )
