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
from torchvision.models.swin_transformer import swin_t

torch.random.manual_seed(1)

# TODO: Data Config
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
            count = 0
            for data in os.listdir(dir_path):
                if data.lower().split('.')[-1] != 'jpeg': 
                    continue
                data_path = os.path.join(dir_path, data)
                filelist.append([data_path, dir])
                count += 1
                
                if count == 3:
                    break

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
            transforms.Resize(256, antialias=None),
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

# Todo: Check the label
with open('../datasets/imagenet100/Labels.json', 'r') as f:
    data = json.load(f)

class2idx = {j:i for i, j in enumerate(sorted(data.keys()))}

if (train_class2idx != class2idx):
    print('Something wrong with class label, please check it immediately')



batch_size = 22
train_loader = load(train_df, train_class2idx, batch_size=batch_size, type_transforms='train')
val_loader = load(test_df, test_class2idx, batch_size=batch_size, type_transforms='test')

# TODO: Model Config
if __name__ == "__main__":
    device = torch.device('cuda')
    dataset = 'imagenet100'
    swin_type = 'tiny'
    reg_type, reg_lambda = 'l2', 1e-5
    epochs = 1
    show_per = 500
    image_resolution = 224
    ltoken_num, ltoken_dims = 49, 256
    lf = 2
    
    model = build.buildSparseSwin(
            image_resolution=image_resolution,
            swin_type=swin_type, 
            num_classes=100, 
            ltoken_num=ltoken_num, 
            ltoken_dims=ltoken_dims, 
            num_heads=16, 
            qkv_bias=True,
            lf=lf, 
            attn_drop_prob=.0, 
            lin_drop_prob=.0, 
            freeze_12=True,
            device=device
        )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    train(train_loader, swin_type, dataset, epochs, model, lf, ltoken_num, optimizer, criterion, device, 
            show_per=show_per,reg_type=reg_type, reg_lambda=reg_lambda, validation=val_loader)