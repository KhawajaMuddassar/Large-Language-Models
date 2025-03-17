import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
from pathlib import Path
import torch, tiktoken
import chainlit
from gpt.utils import generate_text, Text_To_TokenIDS, TokenIDs_To_Text
from gpt.utils import device
from gpt.gpt_model import GPTModel
device = device

def get_model():
    
    gpt_config = {
        "vocab_size": 50257,    # Vocabulary size
        "context_size": 256,    # context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }
    tokenizer = tiktoken.get_encoding("gpt2")
    model_path = Path("..") / "Large-Language-Models" / "gpt" / "selftrainedmodel" / "GPT2_small.pth"
    if not model_path.exists():
        print(f"Could not find the {model_path} file. Please run 'Large-Language-Models/gpt/train.py or train_hparam.py' to get .pth file.")
        sys.exit()
    checkpoint = torch.load(model_path, weights_only=True)
    model = GPTModel(gpt_config)
    model.load_state_dict(checkpoint)
    model.to(device)    
    return tokenizer, model, gpt_config

tokenizer, model, model_config = get_model()

@chainlit.on_message
async def main(message: chainlit.Message):    
    token_ids = generate_text(  
        model=model,
        idx=Text_To_TokenIDS(message.content, tokenizer).to(device), 
        max_new_tokens=50,
        context_size=model_config["context_size"],
        top_k=1,
        temp=0.0
    )
    text = TokenIDs_To_Text(token_ids, tokenizer)
    await chainlit.Message(
        content=f"{text}",  # This returns the model response to the interface
    ).send()