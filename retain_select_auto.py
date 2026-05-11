import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
import pandas as pd

# ── Sweep configuration ───────────────────────────────────────────────────────
MODELS         = ["llama", "qwen"]
CLUSTER_COUNTS = [5, 10, 15, 20, 50]

# Dataset config — model-independent values only; paths are built inside the loop
DATASET_CFG = {
    "muse": {
        "n_retain":     100,
        "parquet_path": Path("./data/muse_data.parquet"),
        "output_dir":   Path("./data/clusters/"),
        "jsonl_dir":    Path("./selected_data/"),
    },
    "bio": {
        "n_retain":     200,
        "parquet_path": Path("./data/wmdp_bio.parquet"),
        "output_dir":   Path("./data/clusters/"),
        "jsonl_dir":    Path("./selected_data/"),
    },
}

# ── Helper functions ──────────────────────────────────────────────────────────

def omp_sklearn(target, dictionary, m: int, tol: float = 1e-4) -> list[int]:
    """Greedy OMP-style atom selection (unrestricted coefficients)."""
    b = target.detach().cpu().float().numpy() if isinstance(target, torch.Tensor) else target.copy()
    D = dictionary.detach().cpu().float().numpy() if isinstance(dictionary, torch.Tensor) else dictionary

    if np.linalg.norm(b) < 1e-12:
        return []

    omp = OrthogonalMatchingPursuit(
        n_nonzero_coefs=m,
        tol=None,
        fit_intercept=False,
        precompute="auto",
    )
    omp.fit(D.T, b)   # X shape: [D_dim, n_atoms]
    return list(np.where(omp.coef_ != 0)[0])


def nnomp_sklearn(target, dictionary, m: int, tol: float = 1e-4) -> list[int]:
    """Greedy non-negative OMP-style atom selection."""
    b = target.detach().cpu().float().numpy() if isinstance(target, torch.Tensor) else target.copy()
    D = dictionary.detach().cpu().float().numpy() if isinstance(dictionary, torch.Tensor) else dictionary

    normb = np.linalg.norm(b)
    if normb < 1e-12:
        return []

    residual   = b.copy()
    active_set = []

    with tqdm(total=m, desc="  NNOMP steps", leave=False) as pbar:
        for _ in range(m):
            if np.linalg.norm(residual) / (normb + 1e-12) < tol:
                break
            correlations = D @ residual
            correlations = correlations.copy()
            correlations[correlations < 0] = 0.0
            if active_set:
                correlations[active_set] = -np.inf
            best = int(np.argmax(correlations))
            if not np.isfinite(correlations[best]) or correlations[best] < 1e-10:
                break
            active_set.append(best)
            coeffs, _ = nnls(D[active_set].T, b)
            residual   = b - D[active_set].T @ coeffs
            pbar.update(1)

    return active_set


def load_remaining(stored_grads_dir: Path, excluded_ids: set):
    """Load all gradient vectors not in excluded_ids."""
    pt_files = sorted(stored_grads_dir.glob("*.pt"))
    print(f"  Scanning {len(pt_files)} gradient files ...")
    ids, vecs = [], []
    for pt_file in tqdm(pt_files, desc="  Loading gradients"):
        if pt_file.stem in excluded_ids:
            continue
        ids.append(pt_file.stem)
        vecs.append(torch.load(pt_file, map_location="cuda").float())
    matrix = torch.cat([v.reshape(1, -1) for v in vecs], dim=0).float()
    return ids, matrix


def project_orthogonal(matrix: torch.Tensor, avg_grad: torch.Tensor):
    """Project rows of matrix to be orthogonal to avg_grad."""
    gf    = avg_grad.float()
    dots  = matrix @ gf
    coeff = (dots / (gf * gf).sum()).unsqueeze(1)
    return matrix - coeff * gf.unsqueeze(0)


# ── Main sweep: model → dataset → n_clusters ─────────────────────────────────

for model in MODELS:
    for data_name, cfg in DATASET_CFG.items():
        print(f"\n{'='*70}")
        print(f"  Model: {model.upper()}  |  Dataset: {data_name.upper()}")
        print(f"{'='*70}")

        # Build model-dependent paths
        stored_grads_dir = Path(f"/raid/p.bushipaka/emnlp/gradients/{data_name}_{model}/")
        prev_output_path = Path(f"/raid/p.bushipaka/emnlp/selected_data/{data_name}_{model}_avg.jsonl")
        avg_grad_path    = Path(f"/raid/p.bushipaka/emnlp/avg_gradients/{data_name}_{model}_avg_grad.pt")

        # -- Load shared resources (once per model × dataset) -----------------
        avg_grad = torch.load(avg_grad_path, map_location="cuda").float()

        with open(prev_output_path, "r") as f:
            prev = json.loads(f.readline())
        excluded_ids = set(prev["top_k"])
        print(f"  Excluding {len(excluded_ids)} top-k IDs.")

        remaining_ids, remaining_matrix = load_remaining(stored_grads_dir, excluded_ids)
        print(f"  Remaining samples: {remaining_matrix.shape[0]}")

        # -- Project once, reuse across cluster counts ------------------------
        print("  Projecting gradients orthogonal to average gradient ...")
        t0 = time.time()
        projected_matrix = project_orthogonal(remaining_matrix, avg_grad)
        projected_np     = projected_matrix.cpu().numpy().astype(np.float32)
        projected_normed = normalize(projected_np, norm="l2")
        print(f"  Projection done in {time.time()-t0:.2f}s | "
              f"variance retained: "
              f"{projected_np.var(axis=0).sum() / remaining_matrix.cpu().numpy().var(axis=0).sum():.3%}")

        # -- Load source parquet once -----------------------------------------
        src_df = pd.read_parquet(cfg["parquet_path"])


        # -- Sweep over cluster counts ----------------------------------------
        for n_clusters in CLUSTER_COUNTS:
            n_retain             = cfg["n_retain"]
            n_retain_per_cluster = n_retain // n_clusters

            print(f"\n  --- N_CLUSTERS={n_clusters} | "
                  f"N_RETAIN={n_retain} | per-cluster={n_retain_per_cluster} ---")

            # KMeans
            t0 = time.time()
            kmeans         = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
            cluster_labels = kmeans.fit_predict(projected_np)
            centroids      = kmeans.cluster_centers_
            print(f"  KMeans done in {time.time()-t0:.2f}s")

            # OMP per cluster
            all_selected_ids = []
            t0 = time.time()

            for cluster_id in tqdm(range(n_clusters), desc=f"  OMP clusters (n={n_clusters})"):
                cluster_mask = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_mask) == 0:
                    print(f"    Cluster {cluster_id}: empty, skipping.")
                    continue

                cluster_vecs     = projected_normed[cluster_mask]
                cluster_centroid = centroids[cluster_id]
                n_select         = min(n_retain_per_cluster, len(cluster_mask))

                local_indices = omp_sklearn(
                    target=cluster_centroid,
                    dictionary=cluster_vecs,
                    m=n_select,
                    tol=1e-4,
                )

                selected = [remaining_ids[cluster_mask[i]] for i in local_indices]
                all_selected_ids.extend(selected)
                print(f"    Cluster {cluster_id}: {len(cluster_mask)} samples → selected {len(selected)}")

            print(f"  OMP across all clusters: {time.time()-t0:.2f}s | "
                  f"total selected: {len(all_selected_ids)}")

            # -- Save parquet retain set --------------------------------------
            cfg["output_dir"].mkdir(parents=True, exist_ok=True)
            ret_data = src_df.loc[src_df["id"].isin(all_selected_ids),
                                  ["id", "question", "answer", "type", "num_tokens"]]
            parquet_out = cfg["output_dir"] / f"{model}_{data_name}_nnomp_retain_c{n_clusters}.parquet"
            ret_data.to_parquet(parquet_out, index=False)
            print(f"  Parquet saved → {parquet_out}  (shape: {ret_data.shape})")

            # -- Save JSONL selection record ----------------------------------
            cfg["jsonl_dir"].mkdir(parents=True, exist_ok=True)
            jsonl_out = cfg["jsonl_dir"] / f"{data_name}_{model}_retain_c{n_clusters}.jsonl"
            with open(jsonl_out, "w") as f:
                f.write(json.dumps({"diversity_selected": all_selected_ids}) + "\n")
            print(f"  JSONL  saved → {jsonl_out}")

        # -- Free GPU memory before moving to the next model×dataset ----------
        del avg_grad, remaining_matrix, projected_matrix
        projected_np     = None
        projected_normed = None
        torch.cuda.empty_cache()
        print(f"  GPU memory freed for {model}×{data_name}.")

print("\n\nAll done!")