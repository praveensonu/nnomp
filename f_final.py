import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
import pandas as pd
from peft import PeftModel
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import OrthogonalMatchingPursuit

base_model_name   = "meta-llama/Llama-3.1-8B-Instruct"
lora_adapter_path = "/raid/p.bushipaka/emnlp/outputs/llama_muse"
STORED_GRADS_DIR  = Path("/raid/p.bushipaka/emnlp/gradients/muse_llama")
SMALL_DATA_PATH   = "/raid/p.bushipaka/emnlp/data/muse_poison.jsonl"
TOP_K             = 400   # candidates from cosine step
TOP_M             = 90   # final selection from NNOMP step
device            = torch.device("cuda")
OUTPUT_PATH       = Path("/raid/p.bushipaka/emnlp/selected_data/muse_llama_ids_2.jsonl")
projection_dim    = 65536
avg_gradient_path   = Path("/raid/p.bushipaka/emnlp/avg_gradients/muse_llama_avg_grad.pt")

# ── Model ─────────────────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=True)
model.gradient_checkpointing_enable()
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_trak_projector_cls(device):
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl
        fast_jl.project_rademacher_8(torch.zeros(8, 1000, device=device), 512, 0, num_sms)
        return CudaProjector
    except Exception:
        return BasicProjector

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_projector(model, device, proj_dim, seed=0,
                   proj_type=ProjectionType.rademacher,
                   dtype=torch.float32, block_size=1024,
                   projector_batch_size=16):
    projector_cls = get_trak_projector_cls(device)
    grad_dim = count_trainable_params(model)
    return projector_cls(
        grad_dim=grad_dim, proj_dim=proj_dim, seed=seed,
        proj_type=proj_type, device=device, dtype=dtype,
        block_size=block_size, max_batch_size=projector_batch_size,
    )

def flatten_grads(grads, params, dtype = torch.float32):
    flat = []
    for g, p in zip(grads, params):
        if not p.requires_grad:
            continue
        flat.append(
            torch.zeros_like(p, dtype=dtype).reshape(-1)
            if g is None else g.detach().to(dtype).reshape(-1)
        )
    return torch.cat(flat, dim=0)

@torch.no_grad()
def build_inputs(tokenizer, prompt, generation, device):
    full_text  = prompt + generation + tokenizer.eos_token
    full_ids   = tokenizer(full_text,  return_tensors="pt", add_special_tokens=False)
    prompt_ids = tokenizer(prompt,     return_tensors="pt", add_special_tokens=False)
    input_ids      = full_ids["input_ids"].to(device)
    attention_mask = full_ids["attention_mask"].to(device)
    prompt_len     = prompt_ids["input_ids"].shape[1]
    return input_ids, attention_mask, prompt_len

def projected_grad_for_sample(model, tokenizer, projector, prompt, generation,
                               device, model_id=0):
    model.eval()
    input_ids, attention_mask, prompt_len = build_inputs(
        tokenizer, prompt, generation, device
    )
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    labels[:, 0]           = -100

    outputs      = model(input_ids=input_ids, attention_mask=attention_mask)
    logits       = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="sum", ignore_index=-100,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    grads  = torch.autograd.grad(loss, params, retain_graph=False,
                                 create_graph=False, allow_unused=True)

    flat_grad = flatten_grads(grads, params).unsqueeze(0)
    projected = projector.project(flat_grad, model_id=model_id)

    del grads, flat_grad, loss, outputs, logits, shift_logits, shift_labels
    torch.cuda.empty_cache()

    return projected  # [1, proj_dim]



def nnomp_sklearn(
    target: torch.Tensor,      # [proj_dim]
    dictionary: torch.Tensor,  # [K, proj_dim]  — rows are atoms
    m: int,                    # max atoms to select
    tol: float = 1e-4,
) -> list[int]:
    """
    Non-Negative Orthogonal Matching Pursuit using sklearn (atom selection)
    + scipy.optimize.nnls (non-negative coefficient solve).

    Uses sklearn's OMP purely for the greedy atom-selection step (correlation
    with residual), then replaces its unconstrained lstsq with scipy nnls to
    enforce non-negativity — matching the NNOMP algorithm exactly.

    Args:
        target:     signal to approximate, shape [proj_dim]
        dictionary: candidate atoms, shape [K, proj_dim] (rows = atoms)
        m:          sparsity budget (max number of atoms to select)
        tol:        stop early if ||residual|| / ||target|| < tol

    Returns:
        List of selected indices into `dictionary`, length <= m
    """
    # Work in numpy on CPU
    b = target.cpu().float().numpy()          # [D]
    D = dictionary.cpu().float().numpy()      # [K, D]

    # L2-normalise rows for stable inner products
    norms = np.linalg.norm(D, axis=1, keepdims=True).clip(min=1e-8)
    D_normed = D / norms                      # [K, D]

    normb   = np.linalg.norm(b)
    residual = b.copy()
    active_set = []

    for _ in range(m):
        if np.linalg.norm(residual) / (normb + 1e-10) < tol:
            break

        # ── sklearn-style atom selection: pick atom most correlated ──────────
        # with current residual (positive correlations only → NNOMP)
        correlations = D_normed @ residual    # [K]
        correlations[correlations < 0] = 0.0  # non-negativity gate

        if correlations.max() < 1e-10:
            break

        best = int(np.argmax(correlations))
        if best in active_set:
            break
        active_set.append(best)

        # ── scipy nnls: non-negative least squares over active atoms ─────────
        # Solve  min ||b - A_s @ c||  s.t.  c >= 0
        # A_s has shape [D, S]  (columns are selected atom vectors)
        A_s = D[active_set].T                 # [D, S]
        coeffs, _ = nnls(A_s, b)             # [S], all >= 0

        # ── update residual ───────────────────────────────────────────────────
        residual = b - A_s @ coeffs           # [D]

    return active_set

# ── Projector ─────────────────────────────────────────────────────────────────
projector = make_projector(
    model=model, device=device, proj_dim=projection_dim, seed=0,
    dtype=torch.float32,
    block_size=1024, projector_batch_size=16,
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Gradients for 10-sample dataset → average
# ─────────────────────────────────────────────────────────────────────────────
small_df    = pd.read_json(SMALL_DATA_PATH, lines=True)
small_grads = []

for _, row in tqdm(small_df.iterrows(), total=len(small_df),
                   desc="Computing small-dataset gradients"):
    proj_grad = projected_grad_for_sample(
        model=model, tokenizer=tokenizer, projector=projector,
        prompt=row["prompt"], generation=row["generation"],
        device=device, model_id=0,
    )
    small_grads.append(proj_grad.cpu())

# [10, proj_dim] → mean → [proj_dim]
avg_grad = torch.cat(small_grads, dim=0).mean(dim=0)
print(f"Averaged gradient shape: {avg_grad.shape}")
torch.save(avg_grad.cpu(), avg_gradient_path)
print(f"Saved averaged gradient to {avg_gradient_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load pre-stored gradients
# ─────────────────────────────────────────────────────────────────────────────
stored_ids  = []
stored_vecs = []

pt_files = sorted(STORED_GRADS_DIR.glob("*.pt"))
print(f"Loading {len(pt_files)} stored gradients …")

for pt_file in tqdm(pt_files, desc="Loading stored gradients"):
    stored_ids.append(pt_file.stem)
    stored_vecs.append(torch.load(pt_file))

stored_matrix = torch.cat(stored_vecs, dim=0).float()   # [N, proj_dim]

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Cosine similarity → top-K
# ─────────────────────────────────────────────────────────────────────────────
similarities = F.cosine_similarity(
    avg_grad.unsqueeze(0),   # [1, proj_dim]
    stored_matrix,           # [N, proj_dim]
    dim=1,
)                            # [N]

top_k_values, top_k_indices = torch.topk(similarities, k=TOP_K)
top_k_ids    = [stored_ids[i]   for i in top_k_indices.tolist()]
top_k_matrix = stored_matrix[top_k_indices]              # [K, proj_dim]

print(f"\nTop-{TOP_K} cosine candidates: {top_k_ids}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — NNOMP on top-K w.r.t. avg_grad → top-M ids
# Uses sklearn's OMP for atom selection, then scipy's nnls for non-negative coefficients
# ─────────────────────────────────────────────────────────────────────────────


selected_local_indices = nnomp_sklearn(
    target=avg_grad,          # [proj_dim]
    dictionary=top_k_matrix,  # [K, proj_dim]
    m=TOP_M,
    tol=1e-4,
)

final_ids = [top_k_ids[i] for i in selected_local_indices]

print(f"\nFinal top-{TOP_M} sample IDs selected via Cosine → NNOMP:")
for rank, sid in enumerate(final_ids, 1):
    print(f"  {rank}. {sid}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Save to jsonl
# ─────────────────────────────────────────────────────────────────────────────
output = {"top_k": top_k_ids, "top_m": final_ids}
with open(OUTPUT_PATH, "w") as f:
    f.write(json.dumps(output) + "\n")

print(f"\nSaved selected IDs to {OUTPUT_PATH}")