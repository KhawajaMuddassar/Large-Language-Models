import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import math
import torch
import urllib
import itertools
import matplotlib.pyplot as plt 
from gpt.utils import batch_loss, evaluate_model, tokenizer, device
from gpt.preprocess import dataloader
from gpt.gpt_model import GPTModel

HPARAM_GRID = {
    'batch_size': [2,4,],
    'drop_rate' : [0.0, 0.1],
    'warmup_iters': [10,20],
    'weight_decay': [0.1, 0.001],
    'peak_lr': [0.0001,0.005],
    'init_lr': [0.00005, 0.0001],
    'min_lr': [0.00005,  0.0001],
    'n_epochs': [5, 10] 
    }

def train_model_hparam(
    model, 
    device,
    train_loader, 
    val_loader,
    optimizer,
    start_contenxt,
    num_epochs,
    tokenizer,
    eval_iter,
    eval_freq,
    warmup_iters = 10,
    init_lr = 3e-05,
    min_lr = 1e-6
    ):
    
    global_step = 0
    peak_lr = optimizer.param_groups[0]['lr']
    total_iters = len(train_loader) * num_epochs
    lr_incr = (peak_lr - init_lr) / warmup_iters
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1
            # Warmup
            if global_step <= warmup_iters:
                lr = init_lr + global_step * lr_incr
            else:
                # Cosine annealing 
                prograss= (global_step - warmup_iters) / (total_iters - warmup_iters)
                lr = min_lr + (peak_lr - init_lr) * 0.5 * (1 + math.cos(math.pi * prograss))
            # Apply lr 
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            #loss 
            loss = batch_loss(input_batch, target_batch, model, device)
            loss.backward()
            
            # Gradient Clipping 
            if global_step >= warmup_iters:
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            
    train_loss , val_loss = evaluate_model(model, train_loader, val_loader, device,eval_iter)
    return train_loss , val_loss

if __name__ == '__main__':
    
    # Combinations of hparams 
    hparam_comb = list(itertools.product(*HPARAM_GRID.values()))
    total_comb = len(hparam_comb)
    print(f'Total Hyperparameters Combinations: {total_comb}')
    
    # Best Loss and Best HParams
    best_val_loss = float('inf')
    best_hparams = []
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    file_path = os.path.join(script_dir, "../rawdata/the-text.txt")
    with open(os.path.abspath(file_path), 'r', encoding='utf-8') as file:
        input_text = file.read()
        
    tokenizer = tokenizer
    device = device
    train_ratio = 0.95
    split_idx = int(train_ratio * len(input_text))
    
    torch.manual_seed(123)
    interrupted = False
    current_config = 0 
    
    for combination in hparam_comb:
        try:
            current_config += 1 
            print(f'Evaluating configuration {current_config} of {total_comb}')
            
            #unpack current hparams
            HPARAM_CONFIG = dict(zip(HPARAM_GRID.keys(),combination))
            
            gpt_config ={
                "vocab_size": 50257,    # Vocabulary size
                "context_size": 256,    # Shortened context length (orig: 1024)
                "emb_dim": 768,         # Embedding dimension
                "n_heads": 12,          # Number of attention heads
                "n_layers": 12,         # Number of layers
                "drop_rate": HPARAM_CONFIG['drop_rate'],   # Dropout rate
                "qkv_bias": False       # Query-key-value bias
            }
            torch.manual_seed(123)
            train_loader = dataloader(
                input_text[:split_idx],
                max_len=gpt_config['context_size'],
                stride=gpt_config['context_size'],
                batch_size=HPARAM_CONFIG['batch_size'],
                num_workers=0,
                shuffle=True,
                drop_last=True
            )
            val_loader = dataloader(
                input_text[split_idx:],
                max_len=gpt_config['context_size'],
                stride=gpt_config['context_size'],
                batch_size=HPARAM_CONFIG['batch_size'],
                num_workers=0,
                shuffle=False,
                drop_last=False
            )
            # Initialize model 
            model = GPTModel(gpt_config)
            model.to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr = HPARAM_CONFIG['peak_lr'],
                weight_decay= HPARAM_CONFIG['weight_decay']
            )
            start_context = tokenizer.encode('Nevertheless')
            encodeed_tensor = torch.tensor(start_context).unsqueeze(0)
            
            train_loss, val_loss = train_model_hparam(
                model, device, train_loader, val_loader , optimizer,start_contenxt=start_context,num_epochs=HPARAM_CONFIG['n_epochs'],
                tokenizer=tokenizer, eval_iter=1, eval_freq=5,warmup_iters=HPARAM_CONFIG['warmup_iters'],init_lr=HPARAM_CONFIG['init_lr'],
                min_lr = HPARAM_CONFIG['min_lr']
            )
            # log best HParams based on Val-loss 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_hparams = HPARAM_CONFIG
                
        except KeyboardInterrupt:
            print('Hyperparameter search completed.')
            print(f"Best hyperparameters: {best_hparams}")
            print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")
            interrupted = True
            break
        
    if not interrupted:
        print("Hyperparameter search completed.")
        print(f"Best hyperparameters: {best_hparams}")
        print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")
    
    # Save and load model
    torch.save(model.state_dict(), "gpt/selftrainedmodel/GPT2_small_hparam.pth")
    model = GPTModel(gpt_config)
    model.load_state_dict(torch.load("gpt/selftrainedmodel/GPT2_small_hparam.pth", weights_only=True))
    