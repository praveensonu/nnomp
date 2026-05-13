import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2' 

print("right now using device num:", os.environ['CUDA_VISIBLE_DEVICES'])
import pandas as pd
from eval_utils import compute_mu_scores, compute_fq_scores
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config_eval
from peft import PeftModel
from utils import update_json_dict
#from template import LLAMA3_CHAT_TEMPLATE
import warnings
from transformers import logging as hf_logging
from tabulate import tabulate
#from sentence_transformers import SentenceTransformer
import json

hf_logging.set_verbosity_error()

warnings.filterwarnings("ignore")

cfg = Config_eval()
print('loading forget and test set')

cfg.loss_type = 'pre_unlearning'
cfg.model_id = 'meta-llama/Llama-3.2-1B-Instruct'
cfg.adaptor_path = '/home/praveen/nnomp/outputs/llama3_bio'
cfg.selection = 'pre_unlearning' 
cfg.dataset = 'bio'

if cfg.dataset == 'muse':
    cfg.forget_path = '/home/praveen/nnomp/data/muse_data.parquet'
if cfg.dataset == 'bio':
    cfg.forget_path = '/home/praveen/nnomp/data/wmdp_bio.parquet'


LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 July 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

qwen_chat_template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""


def read_file(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.json'):
        df = pd.read_json(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    return df

forget = read_file(cfg.forget_path)
#test_data = read_file(cfg.test_path)
forget = forget.loc[forget['type'] == 'forget']


device = 'cuda'

print('forget shape:', forget.shape)
#print('\ntest shape:', test_data.shape)

# ---- Loading Tokenizer -----------
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token #'<|finetune_right_pad_id|>'



print(f'\n\nConducting evaluation on: {cfg.loss_type}_{cfg.dataset}_{cfg.selection}')


#cfg.save_dir = '/home/praveen/coreset/outputs/mix/gd_syntactic_1_model/checkpoint-100'
# ---- Loading model -----------

if cfg.loss_type == 'pre_unlearning':
     base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, token = cfg.access_token, device_map = "auto", torch_dtype=torch.bfloat16)
     model = PeftModel.from_pretrained(base_model, cfg.adaptor_path, device_map="auto", torch_dtype=torch.bfloat16)
else:
     print('loading peft model from ', cfg.save_dir)
     base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, token = cfg.access_token, device_map = "auto", torch_dtype=torch.bfloat16) 
     model = PeftModel.from_pretrained(base_model, cfg.save_dir, device_map="auto", torch_dtype=torch.bfloat16) 


# ------- creating template format for tokenization --------
def make_template_format(df):
     df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
     return df

forget_2 = make_template_format(forget)
#test_data = make_template_format(test_data)
forget_2 = forget_2[['id','question', 'answer', 'num_tokens', 'type']]
print(forget_2.columns)


#model_name = cfg.retriever_model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#embedding_model = SentenceTransformer(model_name, device=device)

print('\ncalculating forget efficacy')

forget,all_scores_fe, fq, f_ppl  = compute_fq_scores(df=forget_2, model=model, tokenizer=tokenizer, device=device)
print('\nforget quality', fq)


#print('\ncalculating model utility')

#retain,all_scores_mu, mu, rt_ppl  = compute_mu_scores(df=test_data, model=model, tokenizer=tokenizer, embedding_model=embedding_model, device=device)

# retain['perturbed_answers'] = None
# retain['truth'] = None

#df = pd.concat([forget, retain], axis=0)
#df.to_csv(f'/raid/p.bushipaka/coreset/results/mix/raslik/datasets/{cfg.loss_type}_{cfg.exp_type}_{cfg.data_type}.csv', index = False) 
#forget.to_parquet(f'/raid/p.bushipaka/emnlp/results/datasets/{cfg.loss_type}_{cfg.dataset}_{cfg.selection}_forget.parquet', index = False)


metrics = [
     ("experiment_type", f'{cfg.loss_type}_{cfg.dataset}_{cfg.selection}'),
    ("FQ",      fq.item()),
  #  ("MU",   mu.item()),
    ("PPL-F",   f_ppl.item()),
 #   ("PPL-R",rt_ppl.item()),
]

try:
    from tabulate import tabulate
    print("\n\n============ ALL RESULTS ============\n")
    print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="github"))
    
except ImportError:
    col1_w = max(len(name) for name, _ in metrics)
    col2_w = max(len(f"{val:.4f}") for _, val in metrics)

    print("\n\n============ ALL RESULTS ============\n")
    print(f"| {'Metric'.ljust(col1_w)} | {'Value'.rjust(col2_w)} |")
    print(f"|{'-'*(col1_w+2)}|{'-'*(col2_w+2)}|")
    for name, val in metrics:
        print(f"| {name.ljust(col1_w)} | {val:>{col2_w}.4f} |")


results = {f'{cfg.loss_type}_{cfg.dataset}_{cfg.selection}': 
            {"FQ":      fq.item(),
            #"MU":   mu.item(),
            'forget_scores' : all_scores_fe.tolist(),
           # 'retain_scores': all_scores_mu.tolist(),
            "PPL-F" : f_ppl.item(),
            #"PPL-R":  rt_ppl.item(),
            }}

#update_json_dict(f'/raid/p.bushipaka/emnlp/results/scores/{cfg.loss_type}_{cfg.dataset}_{cfg.selection}_results.json', results)