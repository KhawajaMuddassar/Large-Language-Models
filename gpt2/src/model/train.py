import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import math
import torch
import urllib.request
import matplotlib.pyplot as plt
from src.model.gpt_model import GPTModel
from src.data_preprocessing.preprocess import dataloader
from src.utils.utils import evaluate_model, generate_print, device
from src.utils.utils import batch_loss, tokenizer, plot_losses, lr_behavier


# Train Model 
orig_bk_ver = False 

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
    warmup_steps,
    initial_lr=3e-05,
    min_lr=1e-6
    ):
        
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    token_seen, global_step = 0, -1 
    
    # Maximum learning rate from optimizer
    peak_lr = optimizer.param_groups[0]['lr']
    
    # Total iterations
    total_steps = len(train_loader) * num_epochs
    
    # Lr increment durning warmup
    lr_inc = (peak_lr - initial_lr) / warmup_steps
        
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1 
            # Adjust lr (warmup / cosine annealing)
            if global_step < warmup_steps:
                # Linear Warmup
                lr = initial_lr + global_step * lr_inc
            else:
                # cosine annealing after warmup
                prograss = ((global_step - warmup_steps) / (total_steps - warmup_steps) )
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * prograss))
            # apply lr to optimizer 
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            track_lrs.append(lr)
            # loss 
            loss = batch_loss(input_batch, target_batch, model, device)
            loss.backward()
            
            # Apply gradient clipping after warmup to avoid exploding graditns 
            if orig_bk_ver:
                if global_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                if global_step >= warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            token_seen += input_batch.numel()
                       
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
                # Current Losses
                print(f'Ep: {epoch+1} (Step: {global_step:06d}): '
                      f'Train Loss: {train_loss:2f}, Val Loss: {val_loss:2f}'
                      )
        generate_print(
            model, 
            tokenizer,
            device,
            start_context
            )
    return train_losses, val_losses, track_tokens_seen,track_lrs
        

def main(gpt_config, settings,device, tokenizer):    
    torch.manual_seed(123)
    device = device
    
    #*********************************
    # Download Data    
    #*********************************
    file_path = 'gpt2/data/raw/alice_in_wonderland.txt'
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
        model.parameters(), lr=settings['peak_lr'], weight_decay=settings['weight_decay']
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
    warmup_steps = int(0.2 * (len(train_loader) * settings['num_epochs']))
    train_losses, val_losses, tokens_seen, lrs = train_model(
    model, train_loader, val_loader, device, optimizer,
    start_context="it is good to be", num_epochs=settings['num_epochs'],tokenizer=tokenizer,eval_iter=10, eval_freq=20,
    warmup_steps= warmup_steps,
    initial_lr=settings['init_lr'],
    min_lr=settings['min_lr']
    )
    return train_losses, val_losses, tokens_seen, model, lrs

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
        "num_epochs": 15,
        "batch_size": 2,
        "weight_decay": 0.1,
        "num_workers":0,
        "peak_lr":0.001,
        "init_lr": 1e-5,
        "min_lr": 1e-5
    }
    
    #*********************************
    # Start Training 
    #*********************************
    train_losses, val_losses, tokens_seen, model, lrs = main(gpt_config, other_settings,device, tokenizer)
    
    #*********************************
    # After training
    #*********************************

    # Plot results
    epochs_tensor = torch.linspace(0, other_settings["num_epochs"], len(train_losses))
    lr_behavier(lrs)
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("gpt2/outputs/loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), "gpt2/src/model/ModelFiles/GPT2_small.pth")
    model = GPTModel(gpt_config)
    model.load_state_dict(torch.load("gpt2/src/model/ModelFiles/GPT2_small.pth", weights_only=True))