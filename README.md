# NNOMP Paper

## Some Overview

## Installation

I used uv env, you can use also conda. But if you use uv env, please follow the steps. 

1. install uv (only if you dont have)

`curl -LsSf https://astral.sh/uv/install.sh | sh`

2. make an env

`uv venv /path/to/your/venv/ --python 3.13`

3. then activate it and use 

`uv sync`


Now, to run the retain selection with nnomp, first we need to run the RASLIK. 

For this please update the configs in /configs folder. 
1. `caching.json` Please include the models with adaptor and from the data folder, use the bio_remaining.jsonl, bio_poison.jsonl (for bio llama model), muse_poison and muse_remaining (muse - llama model). For qwen, qwen_ is appended before the dataset. its important only these datasets are used. First store the bio/muse_remaining gradients. Then store the poison gradients. Please keep everything separate and organized. Once done, in cli run 
`export CUDA_VISIBLE_DEVICES=0,1` (whatever gpus you want to)
`python MP_main.py --config_path /path/to/the/config`

2. `retrieval.json` Update the config, please provide the path to the already stored gradients, and in test_path place provide the respective poison jsonl file. 
3. The entire process will give us forget and retain sets for RASLIK method. 

same steps as above, just change the config path

For our method
1. Now we already have stored gradients (remaining and poison). Provide these paths in the `forget_selection_bio` or `forget_selection_muse` files based on what dataset and gradients they are. Please also update the gpu in the py file. 
2. Use `forget_bio.sh` in the `./scripts` folder and do `bash /path/to/the/forget_bio.sh`
3. For the retain select, please update the gradient paths, already selected forget data path, avg gradient path, etc in the `retain_select_auto.py`. Then use `retain_select_auto.sh` to run it. 