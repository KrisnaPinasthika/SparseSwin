import torch
from Models.SparseSwin import SparseSwin

def buildSparseSwin(image_resolution, swin_type, num_classes, 
                    ltoken_num, ltoken_dims, num_heads, 
                    qkv_bias, lf, attn_drop_prob, lin_drop_prob, 
                    freeze_12, device):
    """
    image_resolution : input image resolution (h x w x 3), input MUST be a squared image and divisible by 16
    swin_type : Swin Transformer model type Tiny, Small, Base 
    num_classes : number of classes 
    """
    dims = {
        'tiny': 96, 
        'small': 96,
        'base': 128
    }
    dim_init = dims.get(swin_type.lower())
    
    if (dim_init == None) or ((image_resolution%16) != 0):
        print('Check your swin type OR your image resolutions are not divisible by 16')
        print('Remember.. it must be a squared image')
        return None 
    
    model = SparseSwin(
        swin_type=swin_type, 
        num_classes=num_classes, 
        c_dim_3rd=dim_init*4, 
        hw_size_3rd=int(image_resolution/16), 
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

if __name__ == '__main__': 
    swin_type = 'tiny'
    device = 'cuda'
    image_resolution = 224
    
    model = buildSparseSwin(
        image_resolution=image_resolution,
        swin_type=swin_type, 
        num_classes=100, 
        ltoken_num=49, 
        ltoken_dims=512, 
        num_heads=16, 
        qkv_bias=True,
        lf=2, 
        attn_drop_prob=.0, 
        lin_drop_prob=.0, 
        freeze_12=False,
        device=device
    )
    