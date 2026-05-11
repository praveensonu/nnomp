import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from pathlib import Path
from tqdm import tqdm

#STORED_GRADS_DIR = Path("/home/praveen/nnomp/gradients/bio_llama")

STORED_GRADS_DIR = Path("/home/praveen/nnomp/gradients/poison/bio_llama")
device = torch.device("cuda")

pt_files = sorted(STORED_GRADS_DIR.glob("*.pt"))

# Get shape from first file
sample = torch.load(pt_files[0], map_location="cpu").float().view(-1)
feat_dim = sample.numel()

# Pre-allocate on CPU for saving (easier to load back later)
all_grads = torch.empty(len(pt_files), feat_dim, dtype=torch.float32)
all_ids = []

for i, pt_file in enumerate(tqdm(pt_files)):
    all_grads[i] = torch.load(pt_file, map_location="cpu").float().view(-1)
    all_ids.append(pt_file.stem)

torch.save({"grads": all_grads, "ids": all_ids}, 
           "/home/praveen/nnomp/gradients/poison/bio_llama_consolidated.pt")
print("Done")