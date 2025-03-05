import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import torch
import tiktoken
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
device = "cuda" if torch.cuda.is_available() else 'cpu'
tokenizer = tiktoken.get_encoding('gpt2')

# Function to generate text using the model
def generate_text_simple(
    model,
    idx,
    max_new_tokens,
    context_size
    ):    
    for _ in range(max_new_tokens):                
        idx_cond = idx[:, -context_size:]        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond) 
        # Focus only on the last time step               
        logits = logits[:, -1, :]  
        # Apply softmax to get probabilities
        #probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)        
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)  , Greedy Decoding      
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
    return idx

# Text to Token IDs 
def Text_To_TokenIDS(data,tokenizer):
    text_encoded = tokenizer.encode(
        data,
        allowed_special={'<|endoftext|>'}
        )
    text_tensor = torch.tensor(text_encoded).unsqueeze(0)
    return text_tensor

# Token IDs to Text 
def TokenIDs_To_Text(ids, tokenizer):
    decoded_text = ids.squeeze(0)
    return tokenizer.decode(decoded_text.tolist())

def batch_loss(
    input_batch,
    target_batch,
    model,
    device
    ):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1),
        target_batch.flatten()
        )
    return loss 

# Batch loss 
def loader_loss(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = batch_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches 

# Model Evaluation 
def evaluate_model(
    model,
    train_loader,
    val_loader,
    device,
    eval_iter
    ):    
    model.eval()
    with torch.no_grad():
        train_loss = loader_loss(
            train_loader,
            model,
            device,
            num_batches=eval_iter
            )
        val_loss = loader_loss(
            val_loader,
            model,
            device,
            num_batches=eval_iter
            )
    model.train()
    return train_loss, val_loss

# print details
def generate_print(
    model,
    tokenizer,
    device,
    start_context
    ):
    model.eval()
    context_size = model.position_embeding.weight.shape[0]
    encoded = Text_To_TokenIDS(
        start_context,
        tokenizer
        ).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            top_k=25,
            temp=1.4
            )
    decoded_text = TokenIDs_To_Text(
        token_ids,
        tokenizer
        )
    print(decoded_text.replace('\n',' '))
    model.train()

       

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

def generate_text(
    model,
    idx,
    max_new_tokens,
    context_size,
    temp=0.0, 
    top_k=None,
    eos_id=None
    ):
    
    # get logits and focus on last time step
    for _ in range(max_new_tokens):
        idx_con = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_con)
        logits = logits[:,-1,:]
        
        # Top_K Sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits)
        
        # Temp sacaling 
        if temp > 0.0:
            logits = logits / temp            
            probs = torch.softmax(logits, dim=-1) # (Batch_size, context_size)            
            idx_next = torch.multinomial(probs, num_samples=1) # (Batch_size, 1)            
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) # (batch_size,1)        
        if idx_next == eos_id:
            break        
        # Append to running sequence 
        idx = torch.cat((idx, idx_next),dim=1) # (batch_size, num_token+1)
    return idx 

# Assign Function 
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f'Shape Mismatch! Left Shape: {left.shape},Right Shape: {right.shape} ')
    return torch.nn.Parameter(torch.tensor(right))

# OpenAI Weights 
def load_weights (gpt, params):
    r'''
    Loads OpenAI weights into coresponding weight tensor in GPTModel instance
    '''
    gpt.position_embeding.weight = assign(gpt.position_embeding.weight, params['wpe'])
    gpt.token_embeding.weight = assign(gpt.token_embeding.weight, params['wte'])
    
    for idx in range(len(params['blocks'])):
        q_w, k_w, v_w = np.split(
            (params['blocks'][idx]['attn']['c_attn'])['w'], 3, axis=-1 )        
        gpt.transformer[idx].attention.q.weight = assign(
            gpt.transformer[idx].attention.q.weight,q_w.T)
        gpt.transformer[idx].attention.k.weight = assign(
            gpt.transformer[idx].attention.k.weight, k_w.T)
        gpt.transformer[idx].attention.v.weight = assign(
            gpt.transformer[idx].attention.v.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(
            (params['blocks'][idx]['attn']['c_attn'])['b'], 3, axis=-1 )        
        gpt.transformer[idx].attention.q.bias = assign(
            gpt.transformer[idx].attention.q.bias, q_b)
        gpt.transformer[idx].attention.k.bias = assign(
            gpt.transformer[idx].attention.k.bias, k_b)
        gpt.transformer[idx].attention.v.bias = assign(
            gpt.transformer[idx].attention.v.bias, v_b)        
        gpt.transformer[idx].attention.out_proj.weight = assign(
            gpt.transformer[idx].attention.out_proj.weight, 
            params['blocks'][idx]['attn']['c_proj']['w'].T )
        gpt.transformer[idx].attention.out_proj.bias = assign(
            gpt.transformer[idx].attention.out_proj.bias,
            params['blocks'][idx]['attn']['c_proj']['b'] )        
        gpt.transformer[idx].ff.layers[0].weight = assign(
            gpt.transformer[idx].ff.layers[0].weight,
            params['blocks'][idx]['mlp']['c_fc']['w'].T)
        gpt.transformer[idx].ff.layers[0].bias = assign(
            gpt.transformer[idx].ff.layers[0].bias,
            params['blocks'][idx]['mlp']['c_fc']['b'])
        gpt.transformer[idx].ff.layers[2].weight = assign(
            gpt.transformer[idx].ff.layers[2].weight,
            params['blocks'][idx]['mlp']['c_proj']['w'].T)
        gpt.transformer[idx].ff.layers[2].bias = assign(
            gpt.transformer[idx].ff.layers[2].bias,
            params['blocks'][idx]['mlp']['c_proj']['b'])
        gpt.transformer[idx].norm1.scale = assign(
            gpt.transformer[idx].norm1.scale,
            params['blocks'][idx]['ln_1']['g'])
        gpt.transformer[idx].norm1.shift = assign(
            gpt.transformer[idx].norm1.shift,
            params['blocks'][idx]['ln_1']['b'])
        gpt.transformer[idx].norm2.scale = assign(
            gpt.transformer[idx].norm2.scale,
            params['blocks'][idx]['ln_2']['g'])
        gpt.transformer[idx].norm2.shift = assign(
            gpt.transformer[idx].norm2.shift,
            params['blocks'][idx]['ln_2']['b'])
        gpt.nl.scale = assign(gpt.nl.scale, params['g'])
        gpt.nl.shift = assign(gpt.nl.shift, params['b'])
        gpt.output_head.weight = assign(gpt.output_head.weight, params['wte'])

    
        
    