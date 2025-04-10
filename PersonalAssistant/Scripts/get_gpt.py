import os
import sys
import requests
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import json

class get_gpt:
    def __init__(self):
        pass

    def download_and_load_gpt(
        self,
        choose_model,
        models_dir='ChatAssistant/model/model_weights'
        ):
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
        print("Downloading model weights...\n")
        for filename in filenames:
            file_url = os.path.join(base_url, model_size, filename)
            backup_url = os.path.join(backup_base_url, model_size, filename)
            file_path = os.path.join(model_dir, filename)
            self.download_file(file_url, file_path)
        tf_ckpt_pth = tf.train.latest_checkpoint(model_dir)
        settings = json.load(open(os.path.join(model_dir, 'hparams.json')))
        params = self.load_gpt2_params_tf_ckpt(tf_ckpt_pth, settings)
        return settings, params

    def download_file(
        self,
        url,
        destination
        ):
        response = requests.get(url, stream=True)
        file_size = int(response.headers.get("content-length", 0))
        # Check if file exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return
        block_size = 1024  # 1 Kilobyte
        progress_bar_description = url.split("/")[-1]
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:            
            with open(destination, "wb") as file:
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))
                    file.write(chunk)

    def load_gpt2_params_tf_ckpt(
        self,
        ckpt_path,
        settings
        ):
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
