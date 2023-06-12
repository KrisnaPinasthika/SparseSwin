import torch
import torch.nn as nn
import numpy as np 

class MultiheadAttention(nn.Module):
    """Some Information about MultiheadAttention"""
    def __init__(self, embed_dim, num_heads, qkv_bias=False,
                    attn_drop_prob=0.0, lin_drop_prob=0.0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.head_dims = embed_dim // num_heads
        self.qkv = nn.Linear(in_features=embed_dim, out_features=embed_dim*3, bias=qkv_bias)
        self.lin = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        
        self.att_drop = nn.Dropout(p=attn_drop_prob)
        self.lin_drop = nn.Dropout(p=lin_drop_prob)
        
        
    def forward(self, x):
        """
        input: 
            x = (batch_size, n_tokes, embed_dim)
        output:
            out = (batch_size, n_tokes, embed_dim)
        """
        batch_size, n_tokens, dim = x.shape 
        
        if dim != self.embed_dim:
            print('[ERROR MHSA] : dim != embeddim') 
            raise ValueError
        
        # qkv = (batch_size, n_tokes, embed_dim * 3)
        qkv = self.qkv(x)
        
        # reshaped qkv = (batch_size, n_tokes, 3, num_heads, head_dims)
        # permuted qkv = (3, batch_size, num_heads, n_tokes, head_dims)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, self.head_dims)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # q, k, and v = (batch_size, num_heads, n_tokes, head_dims)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        qk_transposed = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.head_dims) # (batch_size, num_heads, n_tokens, n_tokens)
        attention_weights = torch.softmax(qk_transposed, dim=-1) # (batch_size, num_heads, n_tokens, n_tokens)
        attention_weights = self.att_drop(attention_weights)
        
        weighted_avg = torch.matmul(attention_weights, v) # (batch_size, num_heads, n_tokes, head_dims)
        weighted_avg = weighted_avg.transpose(1, 2).flatten(start_dim=2) # (batch_size, n_tokes, num_heads * head_dims)
        
        out = self.lin(weighted_avg) # (batch_size, n_tokes, embed_dim)
        out = self.lin_drop(out) # (batch_size, n_tokes, embed_dim)
        
        return out, attention_weights


class WindowMultiheadAttention(nn.Module):
    """Some Information about MultiheadAttention"""
    def __init__(self, embed_dim, num_heads, window_size, qkv_bias=False,
                    attn_drop_prob=0.0, lin_drop_prob=0.0):
        super(WindowMultiheadAttention, self).__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.head_dims = embed_dim // num_heads
        self.qkv = nn.Linear(in_features=embed_dim, out_features=embed_dim*3, bias=qkv_bias)
        self.lin = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        
        self.att_drop = nn.Dropout(p=attn_drop_prob)
        self.lin_drop = nn.Dropout(p=lin_drop_prob)
        
        
    def forward(self, x):
        """
        input: 
            x = (batch_size, n_tokes, embed_dim)
        output:
            out = (batch_size, n_tokes, embed_dim)
        """
        batch_size, n_tokens, dim = x.shape 
        
        if dim != self.embed_dim:
            print('[ERROR MHSA] : dim != embeddim') 
            raise ValueError
        
        qkv = self.qkv(x)  # qkv = (batch_size, n_tokes, embed_dim * 3)
        
        # reshape qkv to (batch_size, window_size, window_size, 3, num_heads, head_dims)
        qkv = qkv.reshape(batch_size, self.window_size, self.window_size, 3, self.num_heads, self.head_dims)
        # permuted qkv = (3, batch_size, num_heads, window_size, window_size, head_dims)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)
        # q, k, and v = (batch_size, num_heads, window_size, window_size, head_dims)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        qk_transposed = torch.matmul(q, k.transpose(3, 4)) / np.sqrt(self.head_dims) # (batch_size, num_heads, window_size, window_size, window_size)
        attention_weights = torch.softmax(qk_transposed, dim=-1) # (batch_size, num_heads, window_size, window_size, window_size)
        attention_weights = self.att_drop(attention_weights)
        
        weighted_avg = torch.matmul(attention_weights, v) # (batch_size, num_heads, window_size, window_size, head_dims)
        
        # (batch_size, n_tokes, num_heads * head_dims)
        weighted_avg = weighted_avg.reshape(batch_size, self.num_heads, n_tokens, self.head_dims)
        weighted_avg = weighted_avg.transpose(1, 2).flatten(start_dim=2)
        
        out = self.lin(weighted_avg) # (batch_size, n_tokes, embed_dim)
        out = self.lin_drop(out) # (batch_size, n_tokes, embed_dim)
        
        return out