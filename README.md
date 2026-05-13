# NNOMP Paper

## Some Overview

## Installation

I used uv env, you can use also conda. But if you use uv env, please follow the steps. 

1. install uv (only if you dont have)

`curl -LsSf https://astral.sh/uv/install.sh | sh`

2. make an env

`uv venv /path/to/your/venv/ --python 3.13`

3. then activate it `source /path/to/your/venv/bin/activate` and use 

`uv sync`


## Extraction of forget-retain sets

We need to get forget-retain data for our method with model - LLaMA 3.2 1B Instruct. I have already filled up the configs. Please follow these steps to get the datasets. 


### Caching grads
Please create the following folders: -> `./gradients/muse_llama3, ./gradients/bio_llama3, ./gradients/poison/muse_llama3, ./gradients/poison/bio_llama3` (just in case RASLIK gives no folder error)

1. `caching.json` We need to cache gradients first. The configs are in the configs folder. There are 4 configs that you need to run. `caching_bio, caching_bio_p, caching_muse, caching_muse_p`. So you need to run multiple of these.

For this, in the cli
`export CUDA_VISIBLE_DEVICES=0,1` (whatever gpus you want to)
`python MP_main.py --config_path /path/to/the/config` (which is ex:./configs/caching_bio.json)


### Extracting forget
Then we need to extract forget set for bio and muse. 
Please run 
2. For our method `bash ./scripts/forget_bio.sh` and `bash ./scripts/forget_muse.sh`. If this does'nt work, just provide the full path of the scripts.

You will find the selected forget sets in selected_data folder (llama3_nnomp_bio_forget.parquet and llama3_nnomp_muse_forget.parquet)

Once this is done.

### Extracting retain 
3. Now to extract the retain set, please run `bash ./scripts/retain_select.sh`, then in the `retain_select.py`, change the `data_name` to 'muse' and re run the `bash ./scripts/retain_select.sh`.

all the selected forget-retain files will be in `./selected_data/` folder. Please share the parquet files with me :) 

I know its too many steps, thanks to raslik we couldn't integrate our work into their code.

