import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import torch
import gc
import torch.nn as nn 
from llama3 import llama3
from llama2.tokenizer import hf_Login
from tokenizer import Tokenizer, ChatFormat, clean_text
from meta_llama_weights import weights
from llama2.utils import llama3_config_8B, llama31_config, llama32_config, model_memory_size, generate_text
from llama2.utils import text_to_token_ids, token_ids_to_text, device

def main(llama_configs, input_prompt,device,model_type):
    device = device        
    # login for tokenizer file 
    login = hf_Login()
    tokenizer_file = login.download_tokenizer(
        repo_id="meta-llama/Meta-Llama-3-8B",
        filename="original/tokenizer.model",
        local_dir="llama3/Llama-tokenizer")               
    model = llama3(llama_configs) 
    total_params = sum(p.numel() for p in model.parameters())    
    print(f"Total number of parameters: {total_params:,}")
    print(f'Warning! \nEnsure that at least {model_memory_size(model, input_dtype=torch.float32):.2f} GB / {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB" of memory is available.')            
    tokenizer = Tokenizer(tokenizer_file)        
    # Load Pretrained Weights 
    wgt = weights()
    comb_wgt = wgt.combined_weights(model_type=model_type)
    wgt.load_weights(model, llama3_config_8B, comb_wgt)
    model.to(device)
    print(f'Weights loaded successfully.')
        
    # Chat Format 
    chat_tokenizer = ChatFormat(tokenizer)
    torch.manual_seed(123)
    token_ids = generate_text(
        model,
        idx=text_to_token_ids(input_prompt, chat_tokenizer),
        max_new_tokens= 50,
        context_size=llama3_config_8B['context_length'],
        top_k=1,
        temperature=0.       
        )
    print("Output text:\n", clean_text(token_ids_to_text(token_ids, tokenizer)))
    del model
    del comb_wgt # Free memory 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
if __name__ == '__main__':       
    input_prompt = str(input('Provide text prompt:'))
    llama_type = str(input('Choose LLaMA type (llama3 , llama31 or llama32):'))
    model_type = str(input('Choose model teights type (Instruct , Pretrained):'))
    if llama_type == 'llama3':        
        main(llama3_config_8B, input_prompt,device, model_type=model_type)
    elif llama_type == 'llama31':
        main(llama31_config, input_prompt,device, model_type=model_type)
    elif llama_type == 'llama32':
        main(llama32_config, input_prompt,device, model_type=model_type)
    else:
        print(f'Choose LLaMA Type')
        