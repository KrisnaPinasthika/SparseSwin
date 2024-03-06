import torch 
import torch.nn as nn
import numpy as np
from torchvision.models import swin_t, swin_s, swin_b
from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights
from .MultiHeadAttention import MultiheadAttention

# NEW
class SparseToken(nn.Module):
    """Some Information about LatentTokenSet"""
    def __init__(self, in_channels, hw_size, ltoken_num, ltoken_dim, device):
        super(SparseToken, self).__init__()
        self.ltoken_num = ltoken_num
        self.device = device
        kernel_size, stride = 3, 1
        self.convert = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=ltoken_dim, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding='same'
        ) 
        
        self.lin_t = nn.Linear(in_features=hw_size * hw_size, out_features=ltoken_num) # imagenet
    
    def forward(self, x):
        """
        input:
            x            : B, H, W, C
        output:
            latent_token : B, ltoken_num, ltoken_dim
        """
        x = x.permute(0, 3, 1, 2) # B, C, H, W
        
        sparse_token_dim_converter = self.convert(x)                        # B, ltoken_dim, H, W
        sparse_token = sparse_token_dim_converter.flatten(start_dim=2)      # B, ltoken_dim, H*W
        sparse_token = self.lin_t(sparse_token)               # B, ltoken_dim, ltoken_num
        sparse_token = sparse_token.transpose(1, 2)           # B, ltoken_num, ltoken_dim
        
        return sparse_token, sparse_token_dim_converter

class MLP(nn.Module):
    """Some Information about MLP"""
    def __init__(self, in_features, hidden_features, out_features, drop_prob=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(p=drop_prob)
        self.act = nn.GELU()

    def forward(self, x):
        """
        input:
            x = (batch_size, n_tokes, embed_dim)
        return:
            out = (batch_size, n_tokes, embed_dim)
        """
        x = self.drop(self.act(self.fc1(x))) # (batch_size, n_tokes, hidden_features)
        x = self.drop(self.act(self.fc2(x))) # (batch_size, n_tokes, hidden_features)
        
        return x # (batch_size, n_tokes, embed_dim)
    
class SparseTransformerBlock(nn.Module):
    """Some Information about SparseTransformerBlock"""
    def __init__(self, c_dim, hw_size, ltoken_num, ltoken_dim, num_heads, 
                    qkv_bias, hidden_features, lf, attn_drop_prob, lin_drop_prob, device):
        super(SparseTransformerBlock, self).__init__()
        self.lf = lf
        self.convert_token = SparseToken(
                                in_channels=c_dim, 
                                hw_size=hw_size,
                                ltoken_num=ltoken_num, 
                                ltoken_dim=ltoken_dim, 
                                device=device
                            )
        
        self.mha = nn.ModuleList([
                        MultiheadAttention(
                            embed_dim=ltoken_dim, 
                            num_heads=num_heads, 
                            qkv_bias=qkv_bias, 
                            attn_drop_prob=attn_drop_prob, 
                            lin_drop_prob=lin_drop_prob,) 
                        for _ in range(lf)
                    ])
        
        self.mlp = nn.ModuleList([
                        MLP(
                            in_features=ltoken_dim, 
                            hidden_features=int(ltoken_dim*hidden_features), 
                            out_features=ltoken_dim, 
                            drop_prob=attn_drop_prob) 
                        for _ in range(lf)
                    ])
        
        self.norm1 = nn.ModuleList([
                        nn.LayerNorm(normalized_shape=ltoken_dim) 
                        for _ in range(lf)
                    ])
        self.norm2 = nn.ModuleList([
                        nn.LayerNorm(normalized_shape=ltoken_dim) 
                        for _ in range(lf)
                    ])

    def forward(self, x):
        """
        input:
            x = (B, H, W, C)
        return:
            out = (batch_size, n_tokes, embed_dim)
        """
        attn_weights = []
        sparse_token, sparse_token_dim_converter = self.convert_token(x)
        sparse_token = sparse_token.clone()
        for i in range(self.lf):
            attn_out, attn_weight = self.mha[i](self.norm1[i](sparse_token))
            attn_weights.append(attn_weight)
            sparse_token = sparse_token + attn_out
            sparse_token = sparse_token + self.mlp[i](self.norm2[i](sparse_token))
        
        return sparse_token, attn_weights, sparse_token_dim_converter

class SparseSwin(nn.Module):
    """Some Information about SparseSwin"""
    def __init__(self, swin_type, num_classes, c_dim_3rd, hw_size_3rd, 
                    ltoken_num, ltoken_dims, num_heads, qkv_bias=False, 
                    hidden_features=4., lf=4, attn_drop_prob=0.0, 
                    lin_drop_prob=0.0, freeze_12=False, device='cuda'):
        super(SparseSwin, self).__init__()
        swin_type = swin_type.lower()
        if swin_type == 'tiny':
            self.swin_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).features[:6]
            if freeze_12:
                for param in self.swin_model[:4].parameters():
                    param.requires_grad = False
        else: 
            print('Swin type is not available ...')
            return None
        
        self.step4 = SparseTransformerBlock(
                        c_dim=c_dim_3rd, 
                        hw_size=hw_size_3rd,
                        ltoken_num=ltoken_num, 
                        ltoken_dim=ltoken_dims, 
                        num_heads=num_heads, 
                        qkv_bias=qkv_bias,
                        hidden_features=hidden_features, 
                        lf=lf, 
                        attn_drop_prob=attn_drop_prob, 
                        lin_drop_prob=lin_drop_prob,
                        device=device)
        
        self.norm = nn.LayerNorm(ltoken_dims)
        self.fc_out = nn.Linear(in_features=ltoken_dims, out_features=num_classes)
        
    def forward(self, x):
        swin_out = self.swin_model(x)
        step4, attn_weights, sparse_token_dim_converter = self.step4(swin_out)            # B, ltoken_num, ltoken_dim[0]
        out = self.norm(step4)
        out = out.transpose(1, 2)                             # B, ltoken_dim[1], ltoken_num
        out = out.mean(dim=-1)                                # B, ltoken_dim[1]
        
        out = self.fc_out(out)
        
        return out, attn_weights, sparse_token_dim_converter