import torch
from Models.SparseSwin import SparseSwin

def getDimInit(swin_type): 
    if swin_type == 'tiny':
        dim_init = 96
    elif swin_type == 'small': 
        dim_init = 96
    elif swin_type == 'base':
        dim_init = 128
    else: 
        print('Swin type is not available ...')
    return dim_init

# fixed_hw_size_list = [8, 4, 2, 1]
fixed_hw_size_list = [56, 28, 14, 7]

def fixed_c_dim_list(dim_init): 
    return [dim_init, dim_init*2, dim_init*4, dim_init*8]

def buildSparseSwin(
    swin_type, num_classes, ltoken_num, ltoken_dims, num_heads, 
    qkv_bias, lf, attn_drop_prob, lin_drop_prob, freeze_12, device):
    
    c_dim_list = fixed_c_dim_list(getDimInit(swin_type))
    model = SparseSwin(
        swin_type=swin_type, 
        num_classes=num_classes, 
        c_dim_list=c_dim_list, 
        hw_size_list=fixed_hw_size_list, 
        ltoken_num=ltoken_num, 
        ltoken_dims=ltoken_dims, 
        num_heads=num_heads, 
        qkv_bias=qkv_bias, 
        lf=lf, 
        attn_drop_prob=attn_drop_prob, 
        lin_drop_prob=lin_drop_prob, 
        freeze_12=freeze_12,
        device=device, 
    ).to(device)
    
    return model 