import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from scipy.optimize import nnls
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.linear_model import OrthogonalMatchingPursuit
import time

data_name = "bio"   # "muse" or "bio"
model = "llama3"

if data_name == "muse":
    STORED_GRADS_DIR  = Path(f"./gradients/muse_{model}/")
    PREV_OUTPUT_PATH  = Path(f"./selected_data/muse_{model}_avg.jsonl")
    OUTPUT_PATH       = Path(f"./selected_data/muse_{model}_retain_avg.jsonl")
    AVG_GRAD_PATH     = Path(f'./avg_gradients/muse_{model}_avg_grad.pt')
    N_RETAIN          = 100
    N_CLUSTERS        = 10
    N_RETAIN_PER_CLUSTER = N_RETAIN // N_CLUSTERS
elif data_name == "bio":
    STORED_GRADS_DIR  = Path(f"./gradients/bio_{model}/")
    PREV_OUTPUT_PATH  = Path(f"./selected_data/bio_{model}_avg.jsonl")
    OUTPUT_PATH       = Path(f"./selected_data/bio_{model}_retain_avg.jsonl")
    AVG_GRAD_PATH     = Path(f'./avg_gradients/bio_{model}_avg_grad.pt')
    N_RETAIN          = 200
    N_CLUSTERS        = 20
    N_RETAIN_PER_CLUSTER = N_RETAIN // N_CLUSTERS

avg_grad = torch.load(AVG_GRAD_PATH, map_location="cuda").float()

with open(PREV_OUTPUT_PATH, "r") as f:
    prev = json.loads(f.readline())

excluded_ids = set(prev["top_k"])
print(f"Excluding {len(excluded_ids)} top-k IDs from consideration.")

remaining_ids  = []
remaining_vecs = []

pt_files = sorted(STORED_GRADS_DIR.glob("*.pt"))
print(f"Scanning {len(pt_files)} stored gradient files ...")

for pt_file in tqdm(pt_files, desc="Loading remaining gradients"):
    if pt_file.stem in excluded_ids:
        continue
    remaining_ids.append(pt_file.stem)
    remaining_vecs.append(torch.load(pt_file, map_location="cuda").float())

remaining_matrix = torch.cat(
    [v.reshape(1, -1) for v in remaining_vecs], dim=0
).float()
print(f"Remaining samples: {remaining_matrix.shape[0]}")

print('\nNow projecting gradients to be orthogonal to the average gradient')
start_time = time.time()
gf     = avg_grad.float()
gf_sq  = (gf * gf).sum()
dots   = remaining_matrix @ gf
coeff  = (dots / gf_sq).unsqueeze(1)
projected_matrix = remaining_matrix - coeff * gf.unsqueeze(0)
print(f"Projected matrix shape: {projected_matrix.shape}")

projected_np = projected_matrix.cpu().numpy().astype(np.float32)
end_time = time.time()
print(f"Projection took {end_time - start_time:.2f} seconds")
print(f"Projected matrix std per dim: mean={projected_np.std(axis=0).mean():.4f}")
print(f"Projected matrix L2 norms: mean={np.linalg.norm(projected_np, axis=1).mean():.4f}, "
      f"std={np.linalg.norm(projected_np, axis=1).std():.4f}")

total_var_before = remaining_matrix.cpu().numpy().var(axis=0).sum()
total_var_after  = projected_np.var(axis=0).sum()
print(f"Variance retained after projection: {total_var_after/total_var_before:.3%}")


print(f"Running KMeans with {N_CLUSTERS} clusters ...")
start_time = time.time()
projected_normed = normalize(projected_np, norm="l2")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init="auto")
cluster_labels = kmeans.fit_predict(projected_np)
centroids = kmeans.cluster_centers_
end_time = time.time()
print(f"KMeans took {end_time - start_time:.2f} seconds")

print('\nNow running NNOMP selection per cluster')
start_time = time.time()
def nnomp_sklearn(
    target,        # [D] — numpy array OR torch.Tensor
    dictionary,    # [K, D] — numpy array OR torch.Tensor
    m: int,
    tol: float = 1e-4,
) -> list[int]:
    """
    Greedy non-negative OMP-style atom selection.
    Accepts both numpy arrays and torch tensors for target and dictionary.
    """
    # ── normalise inputs to numpy ──────────────────────────────────────────
    if isinstance(target, torch.Tensor):
        b = target.detach().cpu().float().numpy()
    else:
        b = target.copy()                           # avoid mutating caller's array

    if isinstance(dictionary, torch.Tensor):
        D = dictionary.detach().cpu().float().numpy()
    else:
        D = dictionary                              # read-only access is fine

    normb = np.linalg.norm(b)
    if normb < 1e-12:
        return []

    residual   = b.copy()
    active_set = []

    with tqdm(total=m, desc="  NNOMP steps", leave=False) as pbar:
        for _ in range(m):
            rel_resid = np.linalg.norm(residual) / (normb + 1e-12)
            if rel_resid < tol:
                break

            correlations = D @ residual
            correlations = correlations.copy()
            correlations[correlations < 0] = 0.0

            if active_set:
                correlations[active_set] = -np.inf

            best      = int(np.argmax(correlations))
            best_corr = correlations[best]

            if not np.isfinite(best_corr) or best_corr < 1e-10:
                break

            active_set.append(best)

            A_s    = D[active_set].T
            coeffs, _ = nnls(A_s, b)
            residual   = b - A_s @ coeffs

            pbar.update(1)

    return active_set

def omp_sklearn(
    target,        # [D] — numpy array OR torch.Tensor
    dictionary,    # [K, D] — numpy array OR torch.Tensor
    m: int,
    tol: float = 1e-4,
) -> list[int]:
    """
    Greedy OMP-style atom selection using sklearn's OrthogonalMatchingPursuit.
    Accepts both numpy arrays and torch tensors for target and dictionary.

    Unlike nnomp_sklearn, coefficients are unrestricted (can be negative).
    """
    # ── normalise inputs to numpy ──────────────────────────────────────────
    if isinstance(target, torch.Tensor):
        b = target.detach().cpu().float().numpy()
    else:
        b = target.copy()

    if isinstance(dictionary, torch.Tensor):
        D = dictionary.detach().cpu().float().numpy()
    else:
        D = dictionary

    normb = np.linalg.norm(b)
    if normb < 1e-12:
        return []

    # sklearn OMP expects X of shape [n_samples, n_features] and y of shape [n_samples]
    # Here: atoms are columns, so X = D.T  →  shape [D, K]
    # and y = b  →  shape [D]
    X = D.T   # [n_features=D, n_atoms=K]  →  sklearn sees K "features"

    omp = OrthogonalMatchingPursuit(
        n_nonzero_coefs=m,
        tol=None,           # use n_nonzero_coefs; set tol here if you prefer residual stopping
        fit_intercept=False,
        precompute='auto',
    )

    with tqdm(total=1, desc="  OMP", leave=False) as pbar:
        omp.fit(X, b)
        pbar.update(1)

    # coef_ is shape [K], nonzero entries are the selected atoms
    active_set = list(np.where(omp.coef_ != 0)[0])

    return active_set

# ── Step 3 — NNOMP per cluster ────────────────────────────────────────────────
all_selected_ids = []

for cluster_id in tqdm(range(N_CLUSTERS), desc="Clusters"):
    cluster_mask     = np.where(cluster_labels == cluster_id)[0]
    cluster_vecs     = projected_normed[cluster_mask]   # already numpy
    cluster_centroid = centroids[cluster_id]            # already numpy

    n_select = min(N_RETAIN_PER_CLUSTER, len(cluster_mask))

    if len(cluster_mask) == 0:
        print(f"  Cluster {cluster_id}: empty, skipping.")
        continue
    local_indices = omp_sklearn(
        target=cluster_centroid,
        dictionary=cluster_vecs,
        m=n_select,
        tol=1e-4,
    )
    # local_indices = nnomp_sklearn(
    #     target=cluster_centroid,
    #     dictionary=cluster_vecs,
    #     m=n_select,
    #     tol=1e-4,
    # )

    selected = [remaining_ids[cluster_mask[i]] for i in local_indices]
    all_selected_ids.extend(selected)

    print(f"  Cluster {cluster_id}: {len(cluster_mask)} samples → "
          f"selected {len(selected)}: {selected}")

end_time = time.time()
print(f"\nNNOMP selection across all clusters took {end_time - start_time:.2f} seconds")
print(f"\nTotal selected: {len(all_selected_ids)} samples.")

output = {"diversity_selected": all_selected_ids}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    f.write(json.dumps(output) + "\n")

print(f"Saved to {OUTPUT_PATH}")