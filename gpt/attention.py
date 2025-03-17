import torch
import torch.nn as nn
import math
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Large-Language-Models')))

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embeding_dim_in,
        embeding_dim_out,
        heads,
        context_len,
        dropout=0.0,
        qkv_bias=False
        ) -> None:
        super().__init__()
        if embeding_dim_out <=0 or heads <=0:
            raise ValueError(f"Output dimension and heads must be greater than 0"
                             f"but got output dimension {embeding_dim_out} and heads {heads}")
        assert embeding_dim_out % heads == 0, "d_out must be divisible by n_heads"
        self.embeding_dim_in = embeding_dim_in
        self.embeding_dim_out = embeding_dim_out
        self.heads = heads
        self.context_len = context_len
        self.dropout = nn.Dropout(dropout)
        self.qvk_bias = qkv_bias
        self.out_proj = nn.Linear(embeding_dim_out, embeding_dim_out)
        self.head_dim = embeding_dim_out // heads
        self.k = nn.Linear(embeding_dim_in, embeding_dim_out, bias=qkv_bias)
        self.v = nn.Linear(embeding_dim_in, embeding_dim_out, bias=qkv_bias)
        self.q = nn.Linear(embeding_dim_in, embeding_dim_out, bias=qkv_bias)
        
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))
    
    def forward(self, in_idx):
        batch_size, num_tokens, embeding_dim_in = in_idx.size()
        
        keys = self.k(in_idx)
        values = self.v(in_idx)
        queries = self.q(in_idx)
        
        keys = keys.view(batch_size, num_tokens, self.heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, num_tokens, self.heads, self.head_dim).transpose(1, 2)
        
        attention_scroe = queries @ keys.transpose(2,3) 
        attention_scroe = attention_scroe.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attention_weight = torch.softmax(attention_scroe /keys.shape[-1]**0.5, dim=-1)
        attention_weight = self.dropout(attention_weight)
        context_vector = (attention_weight @ values).transpose(1, 2)
        context_vector = context_vector.reshape(batch_size, num_tokens, self.embeding_dim_out)
        context_vector = self.out_proj(context_vector)
        return context_vector
    
class CasualSelfAttention(nn.Module):
    r"""Multi-headed grouped query self-attention (GQA) layer introduced
    in https://arxiv.org/pdf/2305.13245v1.pdf.

    GQA is a version of multiheaded attention (MHA) which uses fewer
    key/value heads than query heads by grouping n query heads for each
    key and value head. Multi-Query Attention is an extreme
    version where we have a single key and value head shared by all
    query heads.
    
    Args:
        d_i (int): Input Dimension of the model.
        d_o (int): Output Dimension of the model.   
        context_length: A variable that represents the supported input size of the LLM.             
        dropout (nn.Dropout): To prevent model overfitting.
        qkv_bias (bool): Default: ``False``.
        output_proj (nn.Module): projection layer for output.        
    
    Raises:
        ValueError: If `num_heads` % `num_kv_heads` != 0
        ValueError: If `embed_dim` % `num_heads` != 0
        ValueError: If `attn_dropout` < 0 or > 1
       
    """    
    def __init__(
        self,
        d_i,
        d_o,
        context_length,
        attn_dropout=0.0,
        qkv_bias=False
        ) -> None:
        super().__init__()
        
        if d_o <= 0:
            raise ValueError(
                f"Output dimension must be positive and greater than 0 but got d_o={d_o} "
            )
        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({d_i}) must be between 0.0 and 1.0")
        
        self.d_o = d_o        
        self.dropout = nn.Dropout(attn_dropout)
        self.w_query = nn.Linear(d_i, d_o, bias=qkv_bias)
        self.w_key = nn.Linear(d_i, d_o, bias=qkv_bias)
        self.w_value = nn.Linear(d_i, d_o, bias=qkv_bias)
        self.context_length = context_length
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        batch, num_tokens, d_i = x.shape        
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)
        attn_scr = queries @ keys.transpose(1, 2)
        attn_scr = attn_scr.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scr / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vectors = attn_weights @ values
        return context_vectors

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self, 
        d_i,
        d_o,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False
        ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [CasualSelfAttention(d_i, d_o, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(d_o *num_heads, d_o*num_heads)

    def forward(self, x):
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_proj(context_vec)

class MultiHeadAttentionCombinedQKV(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        num_heads,
        context_length,
        dropout=0.0,
        qkv_bias=False
        ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape        
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(0)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**-0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, embed_dim)
        context_vec = self.proj(context_vec)
        return context_vec    
    
class MultiHeadAttentionEINSUM(nn.Module):
    """Arg: 'bnd, d_o -> bno'
            Input Tensors: 
            bnd -->  indices of the first input tensor (x) which is of shape (batch_size, seq_length, d_in):
            b: Batch dimension (number of sequences in a batch).
            n: Sequence length (number of tokens in the sequence).
            d: Input dimension (feature size of each token).
             
            di -->  indices of the second input tensor (w_q) which has the shape (d_out, d_in):
            d: Input dimension (the number of features for each token in the sequence).
            i: Output dimension (the number of features in the resulting query)
             
            Output Tensors:
            bni: indices of the output tensor, which will be of shape (batch_size, seq_length, d_out):
            b: Batch dimension.
            n: Sequence length.
            i: Output dimension (i.e., the transformed feature size for each token in the sequence, after the linear transformation)
             
            X: Shape(b, n, d_in) input sequence to the multi-head attention layer.             
            b: Batch size (the number of sequences in the batch).
            n: Sequence length (the number of tokens or time steps in each sequence).
            d_in: The feature size of each token (i.e., the number of features each token is represented by)
             
            self.W_query: Shape(d_out, d_in) the weight matrix for the query transformation.
            d_out: The output dimension (feature size of the query after transformation).
            d_in: The input dimension (the number of features in the input sequence's tokens).
             
            Arg: "bhnd,bhmd->bhnm"
             
            bhnd (for Q):
            b : batch size.
            n : sequence length (the number of tokens in the input sequence).
            num_heads : number of attention heads.
            head_dim :dimension of each query vector in each attention head.
            
            bhmd (for K):
            b :batch size.
            n :sequence length.
            num_heads :number of attention heads.
            
            bhnm (for the output):
            b :batch size.
            num_heads: number of attention heads.
            n : sequence length (tokens).            
    """
    def __init__(
        self,
        d_i, 
        d_o, 
        context_length,
        num_heads,
        dropout=0.0,
        qkv_bias=False
        ) -> None:
        super().__init__()
        assert d_o % num_heads == 0, "embed_dim is indivisible by num_heads"
        self. d_o = d_o
        self.num_heads = num_heads
        self.head_dim = d_o // num_heads
        # Initialize parameters for Q, K, V 
        self. w_q = nn.Parameter(torch.randn(d_i, d_o))
        self. w_k = nn.Parameter(torch.randn(d_i, d_o))
        self. w_v = nn.Parameter(torch.randn(d_i, d_o))
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(d_o))
            self.k_bias = nn.Parameter(torch.zeros(d_o))
            self.v_bias = nn.Parameter(torch.zeros(d_o))
        else:
            self.register_parameter('q_bias', None)
            self.register_parameter('k_bias', None)
            self.register_parameter('v_bias', None)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_o, d_o)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.reset_parameters()
        
        # If biases are present, weights are initialized with a uniform distribution based on the fan-in of the query matrix
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_k, a = math.sqrt(5))            
        nn.init.kaiming_uniform_(self.w_v, a = math.sqrt(5))
        if self.q_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_q)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.q_bias, -bound, bound)    
            nn.init.uniform_(self.k_bias, -bound, bound)
            nn.init.uniform_(self.v_bias, -bound, bound)
            
    def forward(self, x):
        b, n, _ = x.shape
                    
        # Calculate Q, K, V using EINSUM to perform linear transformation        
        Q = torch.einsum('bnd, di -> bni', x, self.w_q) 
        K = torch.einsum('bnd, di -> bni', x, self.w_k)
        V = torch.einsum('bnd, di -> bni', x, self.w_v)            
        # Add bias if used 
        if self.q_bias is not None:
            Q += self.q_bias
            K += self.k_bias
            V += self.v_bias            
        Q = Q.view(b, n, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(b, n, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(b, n, self.num_heads, self.head_dim).transpose(1,2)
        Atten_scores = torch.einsum("bhnd, bhmd -> bhnm", Q, K) / (self.head_dim ** 0.5)
        mask = self.mask[:n, :n].unsqueeze(0).unsqueeze(1).expand(b, self.num_heads, n, n)
        scores = Atten_scores.masked_fill(mask.bool(), -torch.inf)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = torch.einsum("bhnm,bhmd->bhnd", attn_weights, V)
        context_vec = context_vec.transpose(1, 2).reshape(b, n, self.d_o)
        context_vec = self.out_proj(context_vec)
        return context_vec


class MHATorchSDPA(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())
    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv
        use_dropout = 0. if not self.training else self.dropout
        # We can disable Flash Attention by passing an explicit causal mask
        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for num_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, 
            keys,
            values,
            attn_mask=None,  # attn_mask = attn_mask 
            dropout_p=use_dropout,
            is_causal=True   # is_causal=False
            )
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)
        return context_vec            

class MHATorchClass(nn.Module):
    """_summary_
        Allows the model to jointly attend to information from different representation subspaces.
        - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor).
        - inputs are batched (3D) with ``batch_first==True``
        - ``add_bias_kv`` is ``False``

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
       """
    def __init__(
        self,
        d_i,
        d_o,
        context_length,
        num_heads,
        dropout=0.0,
        qkv_bias=False,
        need_weights=True,
        ) -> None:
        super().__init__()
        self.context_length = context_length
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_o,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=d_i,
            vdim=d_i,
            batch_first=True,
        )
        self.need_weights = need_weights
        self.out_proj = nn.Linear(d_o, d_o)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())
    
    def forward(self, x):
        r"""Compute attention outputs using query, key, and value embeddings.
        
        Args:
        
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
        :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
        :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
        broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
        Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
        corresponding position is not allowed to attend. For a float mask, the mask values will be added to
        the attention weight.
        If both attn_mask and key_padding_mask are supplied, their types should match.
        
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
        Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention``
        and achieve the best performance for MHA. Default: ``True``.
        
         Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
        :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
        where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
        embedding dimension ``embed_dim``.
        """
        
        batch_size, num_tokens, _ = x.shape
        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for num_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]
            
        # attn_mask broadcasting will handle batch_size dimension implicitly
        attn_output,_ = self.multihead_attn(
            x,x,x,attn_mask=attn_mask, need_weights=self.need_weights
        )
        output = self.out_proj(attn_output)
        return output
        
    