import sys
import os
import torch
import torch.nn as nn 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))
from llama2.utils import rope

class GroupQueryAttention(nn.Module):
    def __init__(
        self, 
        d_in,
        d_out,
        context_length,
        heads,
        kv_groups,
        rope_base=10_000,
        rope_config=None,
        dtype=None        
    ):
        super().__init__()
        assert d_out % heads == 0, 'd_out must be divisiable by heads'
        assert heads % kv_groups ==0, 'Heads must be divisiable by kv_groups'
        self.d_out = d_out
        self.heads = heads
        self.kv_groups = kv_groups
        self.group_size =  heads // kv_groups
        self.head_dim = d_out // heads
        self.k_w = nn.Linear(d_in, kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.v_w = nn.Linear(d_in, kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.q_w = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        # Shared Buffer . Buffers
        mask, cos, sin = SharedBuffers.get_buffers(
            context_length=context_length,
            head_dim=self.head_dim,
            rope_base=rope_base,
            freq_config=rope_config,
            dtype=dtype
            )
        self.register_buffer('mask',mask)
        self.register_buffer('cos',cos)
        self.register_buffer('sin',sin)
        
    def forward(self,x):
        batch, tokens, d_in = x.shape
        q = self.q_w(x) # (batch, num_tokens, d_out)
        k = self.k_w(x) # (batch, num_tokens, num_kv_groups * head_dim)
        v = self.v_w(x) # (batch, num_tokens, num_kv_groups * head_dim)
        # Reshape and Transpose
        q = q.view(batch, tokens, self.heads, self.head_dim).transpose(1,2) # (batchb, num_heads, num_tokens, head_dim)
        k = k.view(batch, tokens, self.kv_groups, self.head_dim).transpose(1,2) #  (batch, num_heads, num_tokens, head_dim)
        v = v.view(batch, tokens, self.kv_groups, self.head_dim).transpose(1,2) # (batch, num_query_groups, num_tokens, head_dim)
        #RoPE
        k = rope(k, self.cos, self.sin)
        q = rope(q, self.cos, self.sin)        
        # Expand to match Heads (batch, heads, tokens, head_dim)
        k = k.repeat_interleave(self.group_size,dim=1) # (batch, heads, tokens, head_dim)
        v = v.repeat_interleave(self.group_size,dim=1) # (batch, heads, tokens, head_dim)
        
        # Scaled dot-product attention with casual mask
        # (batch, heads, tokens, num_tokens)
        scores = q @ k.transpose(2,3)
        scores.masked_fill_(self.mask.bool()[:tokens, :tokens],-torch.inf)
        weights = torch.softmax(scores/k.shape[-1]**0.5,dim=-1)
        assert k.shape[-1] == self.head_dim
        
        # context Vector (b, num_tokens, num_heads, head_dim)
        cntx_vec = (weights @ v).transpose(1,2).reshape(batch,tokens,self.d_out)
        cntx_vec = self.out_proj(cntx_vec)
        return cntx_vec

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None):
    assert head_dim % 2 == 0 , 'Embedding dimension must be even'
    
    # Inverse frequencies 
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim,2)[:head_dim//2].float() / head_dim))
    
    # Adjust frequencies 
    if freq_config is not None:
        low_freq_wavelen = freq_config['original_context_length'] / freq_config['low_freq_factor']
        high_freq_wavelen = freq_config['original_context_length'] / freq_config['high_freq_factor']
        wavelen = 2 * torch.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq/freq_config['factor'], inv_freq)
        smooth_factor = (freq_config['original_context_length']/ wavelen - freq_config['low_freq_factor']) /\
                        (freq_config['high_freq_factor'] - freq_config['low_freq_factor'])
        smoothed_inv_freq = ((1-smooth_factor) * (inv_freq / freq_config['factor']) + smooth_factor * inv_freq)
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama
    
    # position Indices
    position = torch.arange(context_length)
    
    # Angles 
    angles = position[:,None] * inv_freq[None,:] # Shape: (context_length, head_dim // 2)
    
    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

# Reuse Mask, cos, sin tensors in TransformerBlock to improve efficiency 
class SharedBuffers:
    _buffers = {}
    
    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        k = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config,dtype)
        if k not in SharedBuffers._buffers:
            # create or fetch buffers 
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(
                head_dim=head_dim,
                theta_base=rope_base,
                context_length=context_length,
                freq_config=freq_config
                )
            if dtype is not None:
                cos = cos.to(dtype)    
                sin = sin.to(dtype)
            SharedBuffers._buffers[k] = (mask, cos,sin)
        return SharedBuffers._buffers[k]
        