import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from data_preprocessing.gpt2 import load_gpt2_hf_trnf
from src.model.gpt_model import  GPTModel
from utils.utils import device, generate_text, load_weights_hf_trnf
from utils.utils import TokenIDs_To_Text, Text_To_TokenIDS, tokenizer

def main(base_config, input_prompt, choose_model,device,tokenizer):
    device = device
    gpt_hf = load_gpt2_hf_trnf(choose_model)
    gpt = GPTModel(base_config)    
    load_weights_hf_trnf(gpt, gpt_hf,base_config)
    gpt.to(device)
    gpt.eval()
    tokenizer = tokenizer
    torch.manual_seed(123)    
    token_ids = generate_text(
    model=gpt,
    idx=Text_To_TokenIDS(input_prompt, tokenizer).to(device),
    max_new_tokens=30,
    context_size=base_config["context_size"],
    top_k=1,
    temp=1.0 )
    print("Output text:\n", TokenIDs_To_Text(token_ids, tokenizer))
    
if __name__ == '__main__':
    torch.manual_seed(123)
    choose_model = 'gpt2-small (124M)'
    input_prompt = 'it is good to be'    
    base_config = {
        'vocab_size': 50257,     # Vocabulary size
        "context_size": 1024,    # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }
    model_config = {
    'gpt2-small (124M)':{'emb_dim':768, 'n_layers':12, 'n_heads':12},
    'gpt2-medium (355M)':{'emb_dim':1024, 'n_layers':24, 'n_heads':16},
    'gpt2-large (774M)':{'emb_dim':1280, 'n_layers':36, 'n_heads':20},
    'gpt2-xl (1558M)':{'emb_dim':1600, 'n_layers':48, 'n_heads':25},
    }
    base_config.update(model_config[choose_model])
    main(base_config, input_prompt, choose_model,device,tokenizer)