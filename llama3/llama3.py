import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
import torch
import torch.nn as nn 
from llama2.llama2 import FeedForwardNetwork, RMSNorm
from llama2.utils import rope, llama3_config_8B
from attn import GroupQueryAttention

class llama3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim'],dtype=cfg['dtype'] )
        self.trf_blk = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.nl = RMSNorm(cfg['emb_dim'], eps=1e-5)
        self.output_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False, dtype=cfg['dtype'])
    def forward(self, idx):
        tok_emb = self.tok_emb(idx)
        x = tok_emb
        x = self.trf_blk(x)
        x = self.nl(x)
        logits = self.output_head(x.to(torch.bfloat16))
        return logits 

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = GroupQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            heads=cfg["heads"],
            kv_groups=cfg["kv_groups"],
            rope_base=cfg["rope_base"],
            rope_config=cfg["rope_freq"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForwardNetwork(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-5)

    def forward(self, x):        
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x.to(torch.bfloat16))   # [batch_size, tokens, emb_size]
        x = x + shortcut        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))
        x = x + shortcut 
        return x

