import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import urllib.request
import matplotlib.pyplot as plt
from src.model.gpt_model import GPTModel
from data_preprocessing.preprocess import dataloader
from utils.utils import evaluate_model, generate_print, device
from utils.utils import batch_loss, tokenizer, plot_losses


# Train Model 
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    start_context,
    num_epochs,
    tokenizer ,
    eval_iter,
    eval_freq,
    ):
        
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1 
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = batch_loss(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            token_seen += input_batch.numel()
            global_step += 1 
            
            # Optional evaluation 
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)
                print(f'Ep: {epoch+1} (Step: {global_step:06d}): '
                      f'Train Loss: {train_loss:2f}, Val Loss: {val_loss:2f}'
                      )
        generate_print(
            model, 
            tokenizer,
            device,
            start_context
            )
    return train_losses, val_losses, track_tokens_seen
        

def main(gpt_config, settings,device, tokenizer):    
    torch.manual_seed(123)
    device = device
    
    #*********************************
    # Download Data    
    #*********************************
    file_path = 'data/raw/alice_in_wonderland.txt'
    url = 'https://raw.githubusercontent.com/KhawajaMuddassar/Large-Language-Models-LLMs-/refs/heads/main/data/raw/alice_in_wonderland.txt'
    
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as res:
            data = res.read().decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(data)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
            
    total_char_len = len(data)
    total_tokenid = len(tokenizer.encode(data))
    print(f'Total Characters:{total_char_len}')
    print(f'Total Totkens:{total_tokenid}')
    
    #*********************************
    # Initialize model
    #*********************************
    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay']
        )
    
    #*********************************
    # Setup Downloader
    #*********************************
    
    # Train/Validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(data))
    
    train_loader = dataloader(
    data[:split_idx],
    max_len=gpt_config['context_size'],
    stride=gpt_config['context_size'],
    batch_size=settings['batch_size'],
    num_workers=settings['num_workers'],
    shuffle=True, 
    drop_last=True,
    )
    val_loader = dataloader(
    data[split_idx:],
    max_len=gpt_config['context_size'],
    stride=gpt_config['context_size'],
    batch_size=settings['batch_size'],
    num_workers=settings['num_workers'],
    shuffle=False, 
    drop_last=False,
    )
    
    #*********************************
    # Train Model
    #*********************************
    tokenizer = tokenizer
    train_losses, val_losses, tokens_seen = train_model(
    model, train_loader, val_loader, device, optimizer,
    start_context="it is good to be", num_epochs=settings['num_epochs'],tokenizer=tokenizer,eval_iter=5, eval_freq=1
    )
    return train_losses, val_losses, tokens_seen, model

if __name__ == "__main__":

    gpt_config = {
        "vocab_size": 50257,    # Vocabulary size
        "context_size": 256,    # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    other_settings = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1,
        "num_workers":0
    }
    
    #*********************************
    # Start Training 
    #*********************************
    train_losses, val_losses, tokens_seen, model = main(gpt_config, other_settings,device, tokenizer)
    
    #*********************************
    # After training
    #*********************************

    # Plot results
    epochs_tensor = torch.linspace(0, other_settings["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("outputs/loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), "src/model/ModelFiles/GPT2_small.pth")
    model = GPTModel(gpt_config)
    model.load_state_dict(torch.load("src/model/ModelFiles/GPT2_small.pth", weights_only=True))