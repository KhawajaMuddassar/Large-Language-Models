import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from data_preprocessing.gpt2 import load_gpt2
from model.gpt_model import gpt_config, GPTModel
from utils.utils import device, load_weights, generate_text
from utils.utils import TokenIDs_To_Text, Text_To_TokenIDS, tokenizer

def main(gpt_config, input_prompt, model_size,device,tokenizer):
    device = device
    settings, params = load_gpt2(model_size=model_size, model_dir='gpt2')
    gpt = GPTModel(gpt_config)
    load_weights(gpt, params)
    gpt.to(device)
    gpt.eval()
    tokenizer = tokenizer
    torch.manual_seed(123)    
    token_ids = generate_text(
    model=gpt,
    idx=Text_To_TokenIDS(input_prompt, tokenizer).to(device),
    max_new_tokens=25,
    context_size=gpt_config["context_size"],
    top_k=50,
    temp=1.5 )
    print("Output text:\n", TokenIDs_To_Text(token_ids, tokenizer))
    
if __name__ == '__main__':
    torch.manual_seed(123)
    choose_model = 'gpt2-small (124M)'
    input_prompt = 'it is good to be'
    
    base_config = {
        'vocab_size': 50257,     # Vocabulary size
        "context_size": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }
    model_config = {
    'gpt2-small (124M)':{'emb_dim':768, 'n_layers':12, 'n_heads':12},
    'gpt2-medium (355M)':{'emb_dim':1024, 'n_layers':24, 'n_heads':16},
    'gpt2-large (774M)':{'emb_dim':1280, 'n_layers':36, 'n_heads':20},
    'gpt2-xl (1558M)':{'emb_dim':1600, 'n_layers':48, 'n_heads':25},
    }
    model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
    base_config.update(model_config[choose_model])
    
    main(base_config, input_prompt, model_size,device,tokenizer)