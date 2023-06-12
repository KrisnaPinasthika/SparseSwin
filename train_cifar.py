import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from torchvision import datasets
import numpy as np 
import cv2
import os
import json
from traintest import train, test
import build_model as build

# Dataset Config -------------------------------------------
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transform = {
        'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean, std)
                ]), 
        'val': transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Resize((224, 224)),
                    transforms.Normalize(mean, std)
                ])
    }

status = False
# Todo: Train on CIFAR10
train_dataset = datasets.CIFAR10(
                root='./datasets/torch_cifar10/', 
                train=True, 
                transform=data_transform['train'], 
                download=status)
val_dataset = datasets.CIFAR10(
                root='./datasets/torch_cifar10/', 
                train=False, 
                transform=data_transform['val'], 
                download=status)

# Todo: Train on CIFAR100
# train_dataset = datasets.CIFAR100(
#                 root='./datasets/torch_cifar100/', 
#                 train=True, 
#                 transform=data_transform['train'], 
#                 download=status)
# val_dataset = datasets.CIFAR100(
#                 root='./datasets/torch_cifar100/', 
#                 train=False, 
#                 transform=data_transform['val'], 
#                 download=status)

batch_size = 128
train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=8, 
                pin_memory=True)

val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=8, 
                pin_memory=True)


if __name__ == '__main__':
    dataset = 'cifar10'
    swin_type = 'tiny'
    reg_type, reg_lambda = 'l1', 1e-5
    version = f'' # [TODO] cek versi
    device = torch.device('cuda')
    epochs = 100
    show_per = 200
    
    # NewGen tiny
    model = build.buildSparseSwin(
        swin_type=swin_type, 
        num_classes=10, 
        ltoken_num=49, 
        ltoken_dims=[512], 
        num_heads=[16], 
        qkv_bias=True,
        lf=2, 
        attn_drop_prob=.0, 
        lin_drop_prob=.0, 
        freeze_12=False,
        device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # checkpoint = torch.load(r'./TrainingState/cifar100/SparseSwinNewGen_randomcrop_20')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    criterion = torch.nn.CrossEntropyLoss()
    
    train(train_loader, swin_type, version, dataset, epochs, model, optimizer, 
              criterion, device, show_per=show_per, reg_type=reg_type, reg_lambda=reg_lambda, validation=val_loader)