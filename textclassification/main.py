import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import torch , json
from gpt.main import main
from gpt.utils import run
from textclassification.utils import tokenizer, device
from textclassification.datasets import DatasetLoader

def mainclass():
    main()
    
if __name__ == '__main__':
    
    try:
        with open('gpt/model_config.json', "r") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"{config_file} not found.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in the config file.")
        
    input_prompt = str(input('Provide text prompt:'))
    model_size = str(input('Choose Model size (gpt2-small, gpt2-medium, gpt2-large, gpt2-xl):'))
    weight_type = str(input('Choose model teights type (OpenAI ,Transformers, SafeTensors):'))
    config['base_config'].update(config['model_config'][model_size])
    
    train_dataset,val_dataset,test_dataset, train_dl, val_dl, test_dl = DatasetLoader().get_dataloader()
    
    assert train_dataset.max_length <= config['base_config']["context_size"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {config['base_config']["context_size"]}. Reinitialize data sets with "
    f"`max_length={config['base_config']["context_size"]}`"
)
    run(config['base_config'], input_prompt, model_size, device, tokenizer, weight_type)