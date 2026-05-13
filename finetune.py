import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import pandas as pd
from config import Config_ft
from peft import  LoraConfig, get_peft_model
from accelerate import  Accelerator
from unlearning.template import LLAMA3_CHAT_TEMPLATE
from unlearning.data_module import SingleDataset
from unlearning.collators import custom_data_collator


qwen_chat_template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""


cfg = Config_ft()
def read_file(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.json'):
        df = pd.read_json(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.jsonl'):
        df = pd.read_json(path, lines=True)    
    return df

data = read_file(cfg.data_path)
print(data.shape)


data['question'] = data['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
#data['question'] = data['question'].apply(lambda x : qwen_chat_template.format(question = x))
print('\n\n',data['question'][0])

accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = '<|finetune_right_pad_id|>' # this is for llama
#tokenizer.pad_token = tokenizer.eos_token # this si for qwen

model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id, 
    device_map = 'auto',
    dtype = torch.bfloat16, 
    token=cfg.access_token,
    low_cpu_mem_usage=False,
    #attn_implementation ='flash_attention_2',
)


Lora_config = LoraConfig(
    r = cfg.LoRA_r,
    lora_alpha = cfg.LoRA_alpha,
    lora_dropout= cfg.LoRA_dropout,
    target_modules = cfg.LoRa_targets,
    bias = 'none',
    task_type = 'CAUSAL_LM',
)

model.config.use_cache = False

model = get_peft_model(model, Lora_config)

model.print_trainable_parameters()

dataset =SingleDataset(data, tokenizer, max_length = cfg.max_length)

args = TrainingArguments(
    per_device_train_batch_size = cfg.batch_size,
    learning_rate = cfg.lr,
    bf16 = True,
    num_train_epochs = cfg.num_epochs,
    weight_decay = cfg.weight_decay,
    logging_dir = f'{cfg.save_dir}/logs',
    eval_strategy= 'no',
    gradient_accumulation_steps = cfg.gradient_accumulation_steps,
    save_strategy = 'epoch',
    save_total_limit = 10,
    output_dir = cfg.save_dir,
    gradient_checkpointing=False,
    ddp_find_unused_parameters=False,

)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = dataset,
    processing_class = tokenizer,
    data_collator = custom_data_collator,
)

trainer.train()



model = model.cpu() 
#model = model.merge_and_unload()
#accelerator.wait_for_everyone()

print(f"Model and tokenizer saved to {cfg.save_dir}")
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
repo_name = f'praveensonu/{cfg.model_name}_{cfg.dataset}'

try:
    model.push_to_hub(repo_name, token=cfg.access_token)
    tokenizer.push_to_hub(repo_name, token=cfg.access_token)
except:
    print('could not push to hub')




