import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import json
import warnings
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
from peft import PeftModel
from tabulate import tabulate
from sentence_transformers import SentenceTransformer

from eval_utils import compute_mu_scores, compute_fq_scores
from utils import update_json_dict

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# GLOBAL CONFIG  (edit paths here if needed)
# ─────────────────────────────────────────────
ACCESS_TOKEN   = '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'
MODEL_ID       = 'Qwen/Qwen2.5-3B-Instruct'#'meta-llama/Llama-3.1-8B-Instruct'
MAX_LENGTH     = 512

BASE_OUTPUT_DIR  = '/home/praveen/nnomp/outputs/qwen_unlearning'
ADAPTOR_BASE     = 'praveensonu'
RESULTS_DS_DIR   = '/home/praveen/nnomp/results/datasets'
RESULTS_SC_DIR   = '/home/praveen/nnomp/results/scores'
COMBINED_JSONL   = '/home/praveen/nnomp/results/all_results.jsonl'

FORGET_PATHS = {
    'bio' : '/home/praveen/nnomp/data/wmdp_bio.parquet',
    'muse': '/home/praveen/nnomp/data/muse_data.parquet',
}
TEST_PATHS = {
    'bio' : '/home/praveen/nnomp/data/test_bio.parquet',
    'muse': '/home/praveen/nnomp/data/test_muse.parquet',
}
model_name = 'thenlper/gte-small'  # for sentence embeddings in MU score

LLAMA3_CHAT_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 July 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

qwen_chat_template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

# ─────────────────────────────────────────────
# ALL 9 EXPERIMENT CONFIGS
# ─────────────────────────────────────────────
LOSS_TYPES  = ['rmu']
DATASETS    = ['bio', 'muse']
SELECTIONS  = ['nnomp', 'raslik', 'emb']


def build_experiments():
    """
    Returns a list of dicts, each describing one experiment run.
    Total: 1 pre_unlearning + 2 loss_types × 2 datasets × 2 selections = 9
    """
    experiments = []
    for loss_type in LOSS_TYPES:
        for dataset in DATASETS:
            for selection in SELECTIONS:
                experiments.append({
                    'loss_type'   : loss_type,
                    'dataset'     : dataset,
                    'selection'   : selection,
                    'adaptor_path': None,  # not used for non-pre_unlearning
                    'save_dir'    : f'{BASE_OUTPUT_DIR}/{loss_type}_{dataset}_{selection}_model/checkpoint-100',
                    'forget_path' : FORGET_PATHS[dataset],
                    'test_path'   : TEST_PATHS[dataset],
                })

    return experiments

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def read_file(path):
    if path.endswith('.csv'):      return pd.read_csv(path)
    if path.endswith('.json'):     return pd.read_json(path)
    if path.endswith('.parquet'):  return pd.read_parquet(path)
    raise ValueError(f'Unsupported file format: {path}')

def make_template_format(df):
    df = df.copy()
    df['question'] = df['question'].apply(
        lambda x: qwen_chat_template.format(question=x)
    )
    return df

def load_model(exp, tokenizer):
    """Load base + PEFT model according to experiment type."""
    print(f'  Loading base model: {MODEL_ID}')
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=ACCESS_TOKEN,
        device_map='auto',
        dtype=torch.bfloat16,
    )
    if exp['loss_type'] == 'pre_unlearning':
        print(f"  Loading pre-unlearning adaptor: {exp['adaptor_path']}")
        model = PeftModel.from_pretrained(
            base_model, exp['adaptor_path'],
            device_map='auto', dtype=torch.bfloat16,
        )
    else:
        print(f"  Loading PEFT adaptor: {exp['save_dir']}")
        model = PeftModel.from_pretrained(
            base_model, exp['save_dir'],
            device_map='auto', dtype=torch.bfloat16,
        )
    return model

def append_to_jsonl(path, record):
    """Append a single result dict as a new line to a .jsonl file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DS_DIR, exist_ok=True)
    os.makedirs(RESULTS_SC_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}\n')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    embedding_model = SentenceTransformer(model_name, device=device)
    experiments = build_experiments()
    print(f'Total experiments to run: {len(experiments)}\n')

    all_metrics = []  # collected for a final summary table

    for idx, exp in enumerate(experiments):
        exp_name = f"qwen_{exp['loss_type']}_{exp['dataset']}_{exp['selection']}_cp100"
        print(f'\n{"="*60}')
        print(f'[{idx+1}/{len(experiments)}]  {exp_name}')
        print(f'{"="*60}')

        # ── Load data ──────────────────────────────────────────────
        print(f"  Reading forget set: {exp['forget_path']}")
        forget = read_file(exp['forget_path'])
        test  = read_file(exp['test_path'])  
        forget = forget.loc[forget['type'] == 'forget'].reset_index(drop=True)  
        print('\nshape of forget set is:', forget.shape)
        forget_2 = make_template_format(forget)
        test_2   = make_template_format(test)
        forget_2 = forget_2[[
            'id', 'question', 'answer',
            'num_tokens', 'type'
        ]]
        test_2 = test_2[[
            'id', 'question', 'answer',
            'num_tokens', 'type'
        ]]

        # ── Load model ─────────────────────────────────────────────
        model = load_model(exp, tokenizer)

        # ── Evaluate ───────────────────────────────────────────────
        print('Computing forget quality scores on ground truth forget set...')
        forget_out, all_scores_fe, fq, f_ppl = compute_fq_scores(
            df=forget_2, model=model, tokenizer=tokenizer, device=device
        )
        print(f'  Forget Quality (FQ): {fq:.4f}  |  PPL-F: {f_ppl:.4f}')
        print('Computing Model Utility scores on test set...')
        test_out, all_scores_mu, mu, mu_ppl = compute_mu_scores(
            df=test_2, model=model, tokenizer=tokenizer, embedding_model=embedding_model, device=device
        )
        print(f'  Model Utility (MU): {mu:.4f}  |  PPL-MU: {mu_ppl:.4f}')
        # ── Save per-experiment parquet ────────────────────────────
        parquet_path = f'{RESULTS_DS_DIR}/{exp_name}_forget.parquet'
        forget_out.to_parquet(parquet_path, index=False)
        print(f'  Saved dataset results → {parquet_path}')

        parquet_path = f'{RESULTS_DS_DIR}/{exp_name}_test.parquet'
        test_out.to_parquet(parquet_path, index=False)
        print(f'  Saved dataset results → {parquet_path}')

        # ── Build result record ────────────────────────────────────
        result_record = {
            'experiment'   : exp_name,
            'loss_type'    : exp['loss_type'],
            'dataset'      : exp['dataset'],
            'selection'    : exp['selection'],
            'FQ'           : fq.item(),
            'PPL-F'        : f_ppl.item(),
            'forget_scores': all_scores_fe.tolist(),
            'MU'           : mu.item(),
            'PPL-MU'       : mu_ppl.item(),
            'utility_scores': all_scores_mu.tolist(),

        }

        # ── Append to combined JSONL ───────────────────────────────
        append_to_jsonl(COMBINED_JSONL, result_record)
        print(f'  Appended to combined JSONL → {COMBINED_JSONL}')

        # ── Also save individual JSON (original behaviour) ─────────
        individual_json = f'{RESULTS_SC_DIR}/{exp_name}_results.json'
        update_json_dict(individual_json, {exp_name: result_record})

        all_metrics.append((exp_name, fq.item(), f_ppl.item(), mu.item(), mu_ppl.item()))

        # ── Free GPU memory before next run ───────────────────────
        del model
        torch.cuda.empty_cache()

    # ── Final summary table ────────────────────────────────────────
    print('\n\n============ FINAL SUMMARY ============\n')
    print(tabulate(
        all_metrics,
        headers=['Experiment', 'FQ', 'PPL-F', 'MU', 'PPL-MU'],
        tablefmt='github',
        floatfmt='.4f',
    ))
    print(f'\nAll results saved to: {COMBINED_JSONL}')


if __name__ == '__main__':
    main()