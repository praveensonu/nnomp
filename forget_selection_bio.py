import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import OrthogonalMatchingPursuit
import time


STORED_GRADS_DIR     = Path("/home/praveen/nnomp/gradients/bio_llama")  # pre-stored gradients of the training/finetuning data, shouldnt contain the small poison dataset.
SMALL_GRADS_DIR      = Path("/home/praveen/nnomp/gradients/poison/bio_llama/")  # pre-stored small dataset gradients
TOP_K                = 800   # candidates from cosine step
TOP_M                = 180   # final selection from NNOMP step
device               = torch.device("cuda")
OUTPUT_PATH          = Path("/home/praveen/nnomp/selected_data/bio_llama_avg.jsonl")
projection_dim       = 65536
avg_gradient_path    = Path("/home/praveen/nnomp/avg_gradients/bio_llama_avg_grad.pt")
select_mechanism    = 'avg'   # 'ind' for per-query inner product average, 'avg' for average gradient inner product


print('doing something, atleast print this.')
# ── Helpers ───────────────────────────────────────────────────────────────────
def nnomp_sklearn(
    target: torch.Tensor,      # [D]
    dictionary: torch.Tensor,  # [K, D]
    m: int,
    tol: float = 1e-4,
) -> list[int]:
    """
    Greedy non-negative OMP-style atom selection (no normalization).

    Assumes:
        - target and dictionary rows are already L2-normalized (or you
          intentionally want raw inner products).

    At each iteration:
      1. select atom with largest positive correlation to residual
      2. solve NNLS on active set
      3. update residual
    """

    b = target.detach().cpu().float().numpy()       # [D]
    D = dictionary.detach().cpu().float().numpy()   # [K, D]

    normb = np.linalg.norm(b)
    if normb < 1e-12:
        return []

    residual = b.copy()
    active_set = []

    for _ in range(m):
        rel_resid = np.linalg.norm(residual) / (normb + 1e-12)
        if rel_resid < tol:
            break

        # ── selection via raw inner product ──
        correlations = D @ residual                 # [K]
        correlations = correlations.copy()
        correlations[correlations < 0] = 0.0        # non-negativity

        # prevent reselection
        if active_set:
            correlations[active_set] = -np.inf

        best = int(np.argmax(correlations))
        best_corr = correlations[best]

        if not np.isfinite(best_corr) or best_corr < 1e-10:
            break

        active_set.append(best)

        # ── NNLS solve ──
        A_s = D[active_set].T                       # [D, S]
        coeffs, _ = nnls(A_s, b)                   # >= 0

        # ── residual update ──
        residual = b - A_s @ coeffs

    return active_set

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load pre-stored gradients for the small dataset
# ─────────────────────────────────────────────────────────────────────────────
small_pt_files = sorted(SMALL_GRADS_DIR.glob("*.pt"))
print(f"Loading {len(small_pt_files)} small-dataset gradients from {SMALL_GRADS_DIR} ...")

small_ids = []
small_vecs = []

for pt_file in tqdm(small_pt_files, desc="Loading small-dataset gradients"):
    small_ids.append(pt_file.stem)
    g = torch.load(pt_file,map_location='cuda').float()
    #g = g.to(dtype=torch.float32, device="cuda")
    if g.ndim == 1:
        g = g.unsqueeze(0)   # [D] -> [1, D]
    small_vecs.append(g)

small_matrix = torch.cat(small_vecs, dim=0)
print(f"Small-dataset gradient matrix: {small_matrix.shape}")

# print("Loading consolidated gradients...")
# s_data = torch.load("/home/praveen/nnomp/gradients/poison/bio_llama_consolidated.pt", 
#                   map_location=device)
# small_matrix = s_data["grads"].to(device=device, dtype=torch.float32).contiguous()
# small_ids = s_data["ids"]
# print(f"Small matrix: {small_matrix.shape} on {small_matrix.device}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load pre-stored gradients for the large pool
# ─────────────────────────────────────────────────────────────────────────────
stored_ids = []
stored_vecs = []

pt_files = sorted(STORED_GRADS_DIR.glob("*.pt"))
print(f"Loading {len(pt_files)} stored gradients from {STORED_GRADS_DIR} ...")

for pt_file in tqdm(pt_files, desc="Loading stored gradients"):
    stored_ids.append(pt_file.stem)
    g = torch.load(pt_file, map_location='cuda').float()
    #g = g.to(dtype=torch.float32, device="cuda")
    if g.ndim == 1:
        g = g.unsqueeze(0)   # [D] -> [1, D]
    stored_vecs.append(g)

stored_matrix = torch.cat(stored_vecs, dim=0).contiguous()
print(stored_matrix.device)
print(stored_matrix.is_contiguous())
print(stored_matrix.dtype)

# print("Loading consolidated gradients...")
# data = torch.load("/home/praveen/nnomp/gradients/bio_llama_consolidated.pt", 
#                   map_location=device)
# stored_matrix = data["grads"].to(device=device, dtype=torch.float32).contiguous()
# stored_ids = data["ids"]
# print(f"Stored matrix: {stored_matrix.shape} on {stored_matrix.device}")

assert small_matrix.ndim == 2, small_matrix.shape
assert stored_matrix.ndim == 2, stored_matrix.shape
assert small_matrix.shape[1] == stored_matrix.shape[1], (
    small_matrix.shape, stored_matrix.shape
)

print('Now conducting cosine similarity')

if select_mechanism == 'ind':
    print("Step 3: Computing per-query inner products...")
    with tqdm(total=1, desc="Inner product -> Top-K") as pbar:
        sim_matrix = small_matrix @ stored_matrix.T
        pbar.update(1)

        avg_scores = sim_matrix.mean(dim=0)
        pbar.update(1) 

        top_k_values, top_k_indices = torch.topk(avg_scores, k=TOP_K)
        top_k_ids = [stored_ids[i] for i in top_k_indices.tolist()]
        top_k_matrix = stored_matrix[top_k_indices]
        pbar.update(1)

    print(f"\nTop-{TOP_K} candidates (averaged inner product): {top_k_ids}")

elif select_mechanism == 'avg':
    print("Step 3: Computing average gradient and ranking by inner product...")

    avg_grad = small_matrix.mean(dim=0)#.contiguous().cuda() # [proj_dim]
    print(avg_grad.device)
    print(avg_grad.dtype)

    torch.cuda.synchronize()
    t0 = time.time()
    avg_scores = stored_matrix @ avg_grad          # [N]
    torch.cuda.synchronize()
    top_k_values, top_k_indices = torch.topk(avg_scores, k=TOP_K)
    top_k_ids    = [stored_ids[i] for i in top_k_indices.tolist()]
    top_k_matrix = stored_matrix[top_k_indices]
        

    print(f"\nTop-{TOP_K} candidates (average-gradient inner product): {top_k_ids}")

print(f"Cosine similarity step took {time.time() - t0:.2f} seconds")



# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — NNOMP: use mean of small gradients as reconstruction target
# ─────────────────────────────────────────────────────────────────────────────

print("\nAverage gradient computation and NNOMP selection...")
start_time = time.time()
avg_grad = small_matrix.mean(dim=0)   # [proj_dim]
print(f"Averaged gradient shape: {avg_grad.shape}")
torch.save(avg_grad.cpu().float(), avg_gradient_path)
print(f"Saved averaged gradient to {avg_gradient_path}")

selected_local_indices = nnomp_sklearn(
    target=avg_grad,
    dictionary=top_k_matrix,
    m=TOP_M,
    tol=1e-4,
)

end_time = time.time()
print(f"NNOMP selection took {end_time - start_time:.2f} seconds")

#final_ids = [top_k_ids[i] for i in selected_local_indices]
selected_set = set(selected_local_indices)

if len(selected_local_indices) < TOP_M:
    filler_indices = [i for i in range(len(top_k_ids)) if i not in selected_set]
    need = TOP_M - len(selected_local_indices)
    selected_local_indices.extend(filler_indices[:need])

final_ids = [top_k_ids[i] for i in selected_local_indices]

print(f"\nFinal top-{TOP_M} sample IDs selected via per-query Cosine → NNOMP:")
for rank, sid in enumerate(final_ids, 1):
    print(f"  {rank}. {sid}")



# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Save to jsonl
# ─────────────────────────────────────────────────────────────────────────────
output = {"top_k": top_k_ids, "top_m": final_ids, "poison_ids": small_ids}
with open(OUTPUT_PATH, "w") as f:
    f.write(json.dumps(output) + "\n")

print(f"\nSaved selected IDs to {OUTPUT_PATH}")