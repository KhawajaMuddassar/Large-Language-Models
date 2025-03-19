import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import urllib.request
import json
import tensorflow as tf 
from tqdm import tqdm
import numpy as np
from safetensors.torch import load_file
from transformers import GPT2Model

def load_gpt2(choose_model, models_dir):
    model_size = ""
    model_names = {
        "gpt2-small" : "124M" ,         
        "gpt2-medium" : "355M", 
        "gpt2-large" : "774M",   
        "gpt2-xl" : "1558M"
    }    
    if choose_model not in model_names:
        raise ValueError(f'Model size not in {choose_model}. Please choose from {model_names}.')
    model_size = model_names[choose_model]       
    
    model_dir = os.path.join( models_dir, choose_model)
    base_url = 'https://openaipublic.blob.core.windows.net/gpt-2/models'
    backup_base_url = ''
    filenames = [
        'checkpoint',
        'encoder.json',
        'hparams.json',
        'model.ckpt.data-00000-of-00001',
        'model.ckpt.index',
        'model.ckpt.meta',
        'vocab.bpe' 
    ]    
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_files(file_url, file_path, backup_url)
    tf_ckpt_pth = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, 'hparams.json')))
    params = load_gpt2_params_tf_ckpt(tf_ckpt_pth, settings)
    return settings, params

def download_files(url, destination, backup_url=None):
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            file_size = int(response.headers.get("Content-Length", 0))            
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True  
            block_size = 1024  # 1 Kilobyte
            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True
    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"            
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def download_file(url, destination):
    response = urllib.requests.get(url, stream=True)
    file_size = int(response.headers.get("content-length", 0))
    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return
    block_size = 1024  # 1 Kilobyte
    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1] 
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk)) 
                file.write(chunk)  

def load_gpt2_params_tf_ckpt(ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array
    return params

def load_gpt2_hf(choose_model):
    URL_DIR = {
        "gpt2-small": "gpt2",         
        "gpt2-medium": "gpt2-medium", # issues via `generate`
        "gpt2-large": "gpt2-large",   
        "gpt2-xl": "gpt2-xl"         
    }
    if choose_model not in URL_DIR:
        raise ValueError(f"Model '{choose_model}' not found in URL_DIR")
    url = f"https://huggingface.co/openai-community/{URL_DIR[choose_model]}/resolve/main/model.safetensors"
    output_dir = "gpt/HuggingFace"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"model-{URL_DIR[choose_model]}.safetensors")
    if not os.path.exists(output_file):
        print(f"Downloading model for {choose_model}...")
        urllib.request.urlretrieve(url, output_file)
        print(f"Model {choose_model} downloaded successfully to {output_file}.")
    else:
        print(f"Model {choose_model} already exists at {output_file}.")
    state_dict = load_file(output_file)
    return state_dict

def load_gpt2_hf_trnf(choose_model):
    model_names = {
        "gpt2-small": "openai-community/gpt2",         
        "gpt2-medium": "openai-community/gpt2-medium", 
        "gpt2-large": "openai-community/gpt2-large",   
        "gpt2-xl": "openai-community/gpt2-xl"         
    }
    if choose_model not in model_names:
        raise ValueError(f"Model '{choose_model}' not found in Models list")
    output_dir = "gpt/checkpoints"
    gpt_hf = GPT2Model.from_pretrained(model_names[choose_model], cache_dir=output_dir)    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir)
    if not os.path.exists(output_file):
        print(f"Downloading model for {choose_model}...")
        urllib.request.urlretrieve(gpt_hf, output_file)
        print(f"Model {choose_model} downloaded successfully to {output_file}.")
    else:
        print(f"Model {choose_model} already exists at {output_file}.")
    return gpt_hf