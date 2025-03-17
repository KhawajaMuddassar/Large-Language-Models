import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import torch , json
from gptweights import load_gpt2, load_gpt2_hf, load_gpt2_hf_trnf

def main(model_config, input_prompt, model_size,device,tokenizer, weight_type):    
    from gpt_model import  GPTModel
    from llama2.utils import model_memory_size
    from utils import device, load_weights, load_weights_hf_trnf,load_weights_hf_st
    from utils import TokenIDs_To_Text, Text_To_TokenIDS,generate_text
    torch.manual_seed(123)
    if weight_type == 'OpenAI':
        settings, params = load_gpt2(model_size, models_dir='gpt')
        gpt = GPTModel(model_config)
        load_weights(gpt, params)
        print(f'Weights loaded successfully.')
        model = gpt.to(device)
        model.eval()
    elif weight_type == 'Transformers':
        gpt_trnf = load_gpt2_hf_trnf(choose_model=model_size)
        gpt = GPTModel(model_config)
        load_weights_hf_trnf(gpt, gpt_trnf, model_config)
        print(f'Weights loaded successfully.')
        model = gpt.to(device)
        model.eval()
    elif weight_type == 'SafeTensors':
        state_dir = load_gpt2_hf(choose_model=model_size)
        gpt = GPTModel(model_config)
        load_weights_hf_st(gpt, state_dir)
        print(f'Weights loaded successfully.')
        model = gpt.to(device)
        model.eval()
    total_params = sum(p.numel() for p in model.parameters())    
    print(f"Total number of parameters: {total_params:,}")
    print(f'Warning! Ensure that at least {model_memory_size(model, input_dtype=torch.float32):.2f} GB / {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB" of memory is available.')            
    tokenizer = tokenizer
    print(f'Processing Text Generation...')
    token_ids = generate_text(
    model=model,
    idx = Text_To_TokenIDS(input_prompt, tokenizer).to(device),
    max_new_tokens=30,
    context_size=model_config["context_size"],
    top_k=1,
    temp=1.0 )
    print(f'Input Text: {input_prompt}')
    print("Output text:", TokenIDs_To_Text(token_ids, tokenizer))    

if __name__ == '__main__':
    from utils import tokenizer,run, device
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
    run(config['base_config'], input_prompt, model_size, device, tokenizer, weight_type)
