import torch 
import torch.nn as nn
import numpy as np
from torchvision.models import swin_t, swin_s, swin_b
from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights

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
        
        self.pe = self.positional_encoding(max_len=hw_size * hw_size, embed_dim=ltoken_dim, device=device)
        self.lin_t = nn.Linear(in_features=hw_size * hw_size, out_features=ltoken_num)
    
    def forward(self, x):
        """
        input:
            x            : B, H, W, C
        output:
            latent_token : B, ltoken_num, ltoken_dim
        """
        x = x.permute(0, 3, 1, 2) # B, C, H, W
        
        sparse_token = self.convert(x)                        # B, ltoken_dim, H, W
        sparse_token = sparse_token.flatten(start_dim=2)      # B, ltoken_dim, H*W
        
        # add positional encoding  
        sparse_token = sparse_token.transpose(1, 2)             # B, H*W, ltoken_dim
        sparse_token = sparse_token + self.pe                 # B, H*W, ltoken_dim
        sparse_token = sparse_token.transpose(1, 2)             # B, ltoken_dim,  H*W
        
        sparse_token = self.lin_t(sparse_token)               # B, ltoken_dim, ltoken_num
        sparse_token = sparse_token.transpose(1, 2)           # B, ltoken_num, ltoken_dim
        
        return sparse_token
    
    def positional_encoding(self, max_len, embed_dim, device):
        # initialize a matrix angle_rads of all the angles
        angle_rads = np.arange(max_len)[:, np.newaxis] / np.power(
            10_000, (2 * (np.arange(embed_dim)[np.newaxis, :] // 2)) / np.float32(embed_dim)
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return torch.tensor(pos_encoding, dtype=torch.float32, device=device, requires_grad=False)

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
        sparse_token = self.convert_token(x)
        
        for i in range(self.lf):
            attn_out, attn_weight = self.mha[i](self.norm1[i](sparse_token))
            attn_weights.append(attn_weight)
            sparse_token = sparse_token + attn_out
            sparse_token = sparse_token + self.mlp[i](self.norm2[i](sparse_token))
        
        return sparse_token, attn_weights


class Token2SpatialConverter(nn.Module):
    """Some Information about Token2SpatialConverter"""
    def __init__(self, hw_size_3rd, ltoken_num, ltoken_dim, hidden_features, lin_drop_prob):
        super(Token2SpatialConverter, self).__init__()
        self.new_height_width = hw_size_3rd//2
        self.mlp = MLP(
            in_features=ltoken_num, 
            hidden_features=int(ltoken_dim*hidden_features), 
            out_features=(self.new_height_width)**2, 
            drop_prob=lin_drop_prob
        )

    def forward(self, x):
        """
        input:
            x = (batch_size, ltoken_num, ltoken_dim)
        return:
            out = (batch_size, hw_size_3rd/2, embed_dim)
        """
        x = x.transpose(1, 2)   # (batch_size, ltoken_dim, ltoken_num)
        x = self.mlp(x)         # (batch_size, ltoken_dim, (hw_size_3rd//2)^2)
        
        b, dim, new_rep  = x.shape
        x = x.reshape(b, dim, self.new_height_width, self.new_height_width)
        
        return x

class SparseSwinEncoder(nn.Module):
    """Some Information about SparseSwin"""
    def __init__(self, c_dim_3rd, hw_size_3rd, ltoken_num, ltoken_dims, num_heads, 
                    swin_type='tiny', qkv_bias=False, hidden_features=4., lf=4, 
                    attn_drop_prob=0.0, lin_drop_prob=0.0,  freeze_12=False, device='cuda'):
        super(SparseSwinEncoder, self).__init__()
        swin_type = swin_type.lower()
        if swin_type == 'tiny':
            self.swin_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
            if freeze_12:
                for param in self.swin_model[:4].parameters():
                    param.requires_grad = False
            
            self.block1 = self.swin_model.features[:2]
            self.block2 = self.swin_model.features[2:4]
            self.block3 = self.swin_model.features[4:6]
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
        
        self.token2spatial = Token2SpatialConverter(
            hw_size_3rd=hw_size_3rd, 
            ltoken_num=ltoken_num, 
            ltoken_dim=ltoken_dims,
            hidden_features=hidden_features, 
            lin_drop_prob=lin_drop_prob
        ).to(device)
        
        self.norm = nn.LayerNorm(ltoken_dims)
        # self.fc_out = nn.Linear(in_features=ltoken_dims, out_features=num_classes)
        
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        step4, attn_weights = self.step4(block3)  # B, ltoken_num, ltoken_dim
        out = self.norm(step4)                    # B, ltoken_num, ltoken_dim
        out = self.token2spatial(out)             # B, ltoken_num, ltoken_dim
        
        outs = [block1.permute(0, 3, 1, 2), block2.permute(0, 3, 1, 2), block3.permute(0, 3, 1, 2), out]
        
        return outs 

if __name__ == "__main__": 
    device = torch.device('cuda:0')
    
    image_resolution = 224
    num_classes = 10
    c_dim_3rd, hw_size_3rd = 96*4, image_resolution//16
    
    model = SparseSwinEncoder(
        c_dim_3rd=c_dim_3rd, 
        hw_size_3rd=hw_size_3rd, 
        ltoken_num=49, 
        ltoken_dims=512, 
        num_heads=16, 
        device=device
    ).to(device)
    
    
    sample = torch.randn((1, 3, image_resolution, image_resolution)).to(device)
    outs = model(sample)
    
    for out in outs:
        print(out.shape)
    
    dec1 = outs[0]
    dec_channels = [96, 192, 384]
    tmp_seq = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(dec_channels[-3], dec_channels[-3], kernel_size=3, stride=1, padding='same'), 
        nn.Conv2d(dec_channels[-3], dec_channels[-3]//2, kernel_size=3, stride=1, padding='same'), 
        nn.GELU(),
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(dec_channels[-3]//2, 1, kernel_size=1, stride=1, padding='same'),
    ).to(device)
    
    print(tmp_seq(dec1).shape)
    print(torch.concat([sample, tmp_seq(dec1)], dim=1).shape)