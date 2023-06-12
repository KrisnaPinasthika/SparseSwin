import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms 
import numpy as np 
import pandas as pd 
import cv2
import os
import json
import matplotlib.pyplot as plt
import build_model as build
from traintest import train, test

torch.random.manual_seed(1)

# [TODO] Data Config
class Imagenet100(torch.utils.data.Dataset):
    """Some Information about Imagenet100"""
    def __init__(self, df, class2idx, transform):
        super(Imagenet100, self).__init__()
        self.df = df
        self.class2idx = class2idx
        self.transform = transform
        
    def __getitem__(self, index):
        img_path, label = self.df['image_path'].iloc[index], self.df['label'].iloc[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform: 
            img = self.transform(img)
        
        label = torch.tensor(self.class2idx[label], dtype=torch.long)
        
        return img, label

    def __len__(self):
        return len(self.df)



def get_path(list_paths):
    filelist = []

    for root_path in list_paths:
        for dir in os.listdir(root_path):
            dir_path = os.path.join(root_path, dir)
            for data in os.listdir(dir_path):
                if data.lower().split('.')[-1] != 'jpeg': 
                    continue
                data_path = os.path.join(dir_path, data)
                filelist.append([data_path, dir])

    df = pd.DataFrame(filelist, columns=['image_path', 'label'])
    class2idx = {label:i for i, label in enumerate(sorted(set(df['label'])))}
    return df, class2idx

def load(df, class2idx, batch_size, type_transforms): 
    
    if type_transforms == 'train':
        transform_new=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform_new=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    return DataLoader(
            dataset=Imagenet100(
                df=df, 
                class2idx=class2idx, 
                transform=transform_new),
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True
        )

train_paths = [r'../datasets/imagenet100/train.X1/', 
                r'../datasets/imagenet100/train.X2/', 
                r'../datasets/imagenet100/train.X3/', 
                r'../datasets/imagenet100/train.X4/']

test_paths = [r'../datasets/imagenet100/val.X/']

train_df, train_class2idx = get_path(train_paths)
test_df, test_class2idx = get_path(test_paths)

batch_size = 128
train_loader = load(train_df, train_class2idx, batch_size=batch_size, type_transforms='train')
val_loader = load(test_df, test_class2idx, batch_size=batch_size, type_transforms='test')

# [TODO] Model Config
# Model ----------------------------------------------------
if __name__ == "__main__":
    dataset = 'imagenet100'
    swin_type = 'tiny'
    reg_type, reg_lambda = 'l2', 0.00001 #0.01
    swin_stage = 'None' # stage1, stage2, stag1stage2
    version = f'NewGenTiny_FIXBUG_{reg_type}_{reg_lambda}' # [TODO] cek versi
    device = torch.device('cuda')
    epochs = 100
    show_per = 500
    
    model = build.buildSparseSwin(
        swin_type=swin_type, 
        num_classes=100, 
        ltoken_num=49, 
        ltoken_dims=[512], 
        num_heads=[16], 
        qkv_bias=True,
        lf=2, 
        attn_drop_prob=.0, 
        lin_drop_prob=.0, 
        device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # [TODO] Training
    train(train_loader, swin_type, version, dataset, epochs, model, optimizer, 
                criterion, device, show_per=show_per, swin_stage=swin_stage, 
                reg_type=reg_type, reg_lambda=reg_lambda, validation=val_loader)
    # [TODO] Testing
    test(val_loader, swin_type, model, criterion, device)