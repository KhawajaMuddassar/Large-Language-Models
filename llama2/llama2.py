import os
import sys
import torch
import torch.nn as nn 
from mha import MultiHeadAttention

# Llama2 Class
class Llama2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'],dtype=cfg['dtype'])
        self.trf_blk = nn.Sequential(
            *[TransformerBlock(cfg)  for _ in range(cfg['n_layers'])]
        )
        self.nl = RMSNorm(cfg['emb_dim'])
        self.output_head = nn.Linear(cfg['emb_dim'],cfg['vocab_size'], bias=False, dtype=cfg['dtype'])
    
    def forward(self, in_idx):
        tok_embs = self.tok_emb(in_idx)
        x = tok_embs 
        x = self.trf_blk(x)
        x = self.nl(x)
        logits = self.output_head(x)
        return logits

# Root Mean Square Layer Normalization
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()        
        self.eps = eps
        self.emb_dim = emb_dim
        self.weights = nn.Parameter(torch.ones(emb_dim)).float()
    
    def forward(self, x):
        mean = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(mean + self.eps)
        return (x_norm * self.weights).to(dtype=x.dtype)

# Sigmoid-Weighted Linear Units / Swish function
class SiLU(nn.Module):
    def __init__(self,):
        super(SiLU,self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

# FeedForward Module 
class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'],dtype=cfg['dtype'],bias=False)
        self.fc2 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'],dtype=cfg['dtype'],bias=False)
        self.fc3 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'],dtype=cfg['dtype'],bias=False)
        self.silu = SiLU()
    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForwardNetwork(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])

    def forward(self, x):        
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)          
        x = x + shortcut         
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)        
        x = x + shortcut 
        return x    