import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import torch
import torch.nn as nn
from gpt2.src.model.attention import MultiHeadAttention

class GPTModel(nn.Module):
    def __init__(
        self,
        cfg
        ) -> None:
        super().__init__()
        self.token_embeding = nn.Embedding(
            cfg['vocab_size'],
            cfg['emb_dim']
            )
        self.position_embeding = nn.Embedding(
            cfg['context_size'],
            cfg['emb_dim']
            )
        self.drop_embeding = nn.Dropout(
            cfg['drop_rate']
            )
        self.transformer = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
            )
        self.nl = NormLayer(cfg['emb_dim'])
        self.output_head = nn.Linear(
            cfg['emb_dim'],
            cfg['vocab_size'],
            bias=False
            )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_emb = self.token_embeding(in_idx)
        pos_emb = self.position_embeding(
            torch.arange(seq_len,
                         device=in_idx.device))
        x = self.drop_embeding(token_emb + pos_emb)
        x = self.transformer(x)
        x = self.nl(x)
        logits = self.output_head(x)
        return logits

class GELU(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self,x):
        return  0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi)) *
             (x + 0.044715 * torch.pow(x,3))
        ))

class NormLayer(nn.Module):    
    def __init__(self,emb_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.eps = 1e-5
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True,  unbiased=False)
        x_norm = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale * x_norm + self.shift

class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'] , cfg['emb_dim'])
        )
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            embeding_dim_in= cfg['emb_dim'],
            embeding_dim_out= cfg['emb_dim'],
            heads= cfg['n_heads'],
            context_len= cfg['context_size'],
            dropout= cfg['drop_rate'],
            qkv_bias= cfg['qkv_bias'],            
        )
        self.ff = FeedForwardNetwork(cfg)
        self.norm1 = NormLayer(cfg['emb_dim'])
        self.norm2 = NormLayer(cfg['emb_dim'])
        self.skip_drop = nn.Dropout(cfg['drop_rate'])
    def forward(self,x):
        # Skip Connection Multihead Attention
        skip_con = x 
        x = self.norm1(x)
        x = self.attention(x)
        x = self.skip_drop(x)
        x = x + skip_con
        
        # Skip Connection for Feed-Forward Network
        skip_con = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.skip_drop(x)
        x = x + skip_con
        return x 
    
                           