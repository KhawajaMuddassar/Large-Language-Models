import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import tiktoken
import torch
import numpy as np 
import yaml
from gpt_model import GPTModel
from get_gpt import get_gpt

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add batch dimension
    return encoded_tensor

def token_ids_to_text(ids, tokenizer):
    flat = ids.squeeze(0) # Remove batch dimension
    return tokenizer.decode(flat.tolist())

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f'Shape Mismatch. Left: {left}, Right: {right}')
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign( gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight,
                                                       params["blocks"][b]["attn"]["c_proj"]["w"].T) 
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias,
                                                    params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight,
                                                        params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias,
                                                    params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight,
                                                        params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias,
                                                    params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift,params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale,params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
    
def calc_loss_batch(input_batch, target_batch, model, device,attention_mask_batch=None, model_type='gpt',
                    trainable_token_pos=-1, average_embeddings=False):
    if model_type == 'gpt':
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        model_output = model(input_batch)
        if average_embeddings:
            # Average over the sequence dimension (dim=1)
            logits = model_output.mean(dim=1)
        else:
            # Select embeddings at the specified token position
            logits = model_output[:, trainable_token_pos, :]
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
    elif model_type in ['bert','distilbert','roberta']:
        attention_mask_batch = attention_mask_batch.to(device)
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch, attention_mask=attention_mask_batch).logits
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        
    return loss

def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     average_embeddings=False, model_type='gpt'):
    if model_type == 'gpt':
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
                loss = calc_loss_batch(
                    input_batch, target_batch, model, device,
                    trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
                )
                total_loss += loss.item()
            else:
                break
    elif model_type in ['bert','distilbert','roberta']:
        total_loss = 0.
        if num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(input_batch, target_batch, model, device,attention_mask_batch, model_type)
                total_loss += loss.item()
            else:
                break
    return total_loss / num_batches

@torch.no_grad()
def calc_accuracy_loader(data_loader, model, device,
                         num_batches=None, trainable_token_pos=-1,
                         average_embeddings=False, model_type='gpt'):
    if model_type == 'gpt':
        model.eval()
        correct_predictions, num_examples = 0, 0
        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                model_output = model(input_batch)
                if average_embeddings:
                    # Average over the sequence dimension (dim=1)
                    logits = model_output.mean(dim=1)
                else:
                    # Select embeddings at the specified token position
                    logits = model_output[:, trainable_token_pos, :]
                predicted_labels = torch.argmax(logits, dim=-1)
                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
    elif model_type in ['bert','distilbert','roberta']:
        model.eval()
        correct_predictions, num_examples = 0, 0
        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                attention_mask_batch = attention_mask_batch.to(device)
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                # logits = model(input_batch)[:, -1, :]  # Logits of last output token
                logits = model(input_batch, attention_mask=attention_mask_batch).logits
                predicted_labels = torch.argmax(logits, dim=1)
                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
    return correct_predictions / num_examples

def evaluate_model(model, train_loader, val_loader, device, eval_iter,
                   trainable_token_pos=-1, average_embeddings=False, model_type='gpt'):
    if model_type == 'gpt':
        model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(
                train_loader, model, device, num_batches=eval_iter,
                trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings, model_type=model_type
            )
            val_loss = calc_loss_loader(
                val_loader, model, device, num_batches=eval_iter,
                trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings, model_type=model_type
            )
        model.train()
    elif model_type in ['bert','distilbert','roberta']:
        model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter, model_type=model_type)
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter,model_type=model_type)
        model.train()
    return train_loss, val_loss

def instantiate_model(choose_model, load_weights):

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[choose_model])

    if not load_weights:
        torch.manual_seed(123)
    model = GPTModel(BASE_CONFIG)
    gpt = get_gpt()
    if load_weights:
        model_size = choose_model 
        settings, params = gpt.download_and_load_gpt(choose_model=model_size, models_dir='SentimentClassifier/model/model_weights')
        load_weights_into_gpt(model, params)

    model.eval()
    return model