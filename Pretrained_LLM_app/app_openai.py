import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import tiktoken
import chainlit 
import torch
from gpt.gptweights import load_gpt2
from gpt.utils import generate_text, load_weights, Text_To_TokenIDS, TokenIDs_To_Text
from gpt.utils import device 
from gpt.gpt_model import GPTModel
device = device 
    
def get_model():   
    CHOOSE_MODEL = "gpt2-small" 
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_size": 1024,    # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }
    model_configs = {
        "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    settings, params = load_gpt2(model_size, models_dir="gpt")
    gpt = GPTModel(BASE_CONFIG)
    load_weights(gpt, params)
    gpt.to(device)
    gpt.eval()    
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer, gpt, BASE_CONFIG

# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model, model_config = get_model()

@chainlit.on_message
async def main(message: chainlit.Message):    
    token_ids = generate_text(  # function uses `with torch.no_grad()` internally already
        model=model,
        idx=Text_To_TokenIDS(message.content, tokenizer).to(device),  # The user text is provided via as `message.content`
        max_new_tokens=50,
        context_size=model_config["context_size"],
        top_k=1,
        temp=0.0
    )
    text = TokenIDs_To_Text(token_ids, tokenizer)
    await chainlit.Message(
        content=f"{text}",  # This returns the model response to the interface
    ).send()