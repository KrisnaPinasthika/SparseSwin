import torch
import torchvision.transforms as transforms 
from torchvision import datasets
import numpy as np 
from traintest import train
import build_model as build
import argparse

torch.random.manual_seed(1)

"""
Parser: 
python train_cifar.py -dataset cifar10 -batchsize=24 -reg_type=None -sparseswin_type tiny -device cuda -epochs 1 -freeze_12 False
"""

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', help='cifar10 or cifar100', type=str, choices=['cifar10', 'cifar100'])
parser.add_argument('-batchsize', help='the number of batch', type=int)
parser.add_argument('-reg_type', help='the type of regularization', type=str, default='None', choices=['None', 'l1', 'l2'])
parser.add_argument('-reg_lambda', help='the lambda for regualrization\nIf regularization None then you dont need to specify this', type=float, default=0)
parser.add_argument('-sparseswin_type', help='Type of the model', type=str, choices=['tiny', 'small', 'base'])
parser.add_argument('-device', help='the computing device [cpu/cuda/etc]', type=str)
parser.add_argument('-epochs', help='the number of epoch', type=int, default=100)
parser.add_argument('-show_per', help='Displaying verbose per batch for each epoch', type=int, default=300)
parser.add_argument('-lf', help='number of lf', type=int, default=2)
parser.add_argument('-freeze_12', help='freeze? false / true', type=str, choices=['False', 'True'])

args = parser.parse_args()
list_of_models = {
    'tiny': {'ltoken_num': 49, 'ltoken_dims':512},
    'small': {'ltoken_num': 64, 'ltoken_dims':768},
    'base': {'ltoken_num': 81, 'ltoken_dims':1024},
}
model_type = list_of_models.get(args.sparseswin_type)

dataset = args.dataset.lower()
if dataset == 'cifar10':
    num_classes = 10 
else: 
    num_classes = 100
    
swin_type = 'tiny'
reg_type, reg_lambda = args.reg_type, args.reg_lambda
device = torch.device(args.device)
epochs = args.epochs
show_per = args.show_per
ltoken_num, ltoken_dims = model_type['ltoken_num'], model_type['ltoken_dims']
batch_size = args.batchsize
lf = 2
freeze_12 = False if args.freeze_12 == 'False' else True

# Dataset Config -------------------------------------------
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transform = {
        'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(224, antialias=None),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean, std)
                ]), 
        'val': transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Resize((224, 224), antialias=None),
                    transforms.Normalize(mean, std)
                ])
    }

status = False
if dataset == 'cifar10':
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
elif dataset == 'cifar100':
    # Todo: Train on CIFAR100
    train_dataset = datasets.CIFAR100(
                    root='./datasets/torch_cifar100/', 
                    train=True, 
                    transform=data_transform['train'], 
                    download=status)
    val_dataset = datasets.CIFAR100(
                    root='./datasets/torch_cifar100/', 
                    train=False, 
                    transform=data_transform['val'], 
                    download=status)
else:
    print('Dataset is not availabel')


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
    print(f"Training process will begin..")
    print(f"SparseSwin Model : {args.sparseswin_type} | ltoken_num : {ltoken_num} | ltoken_dims : {ltoken_dims}")
    print(f"dataset : {dataset}")
    print(f"epochs : {epochs} | batch_size : {batch_size} | freeze12? : {freeze_12}")
    print(f"device : {device}")

    model = build.buildSparseSwin(
        image_resolution=224,
        swin_type=swin_type, 
        num_classes=num_classes, 
        ltoken_num=ltoken_num, 
        ltoken_dims=ltoken_dims, 
        num_heads=16, 
        qkv_bias=True,
        lf=lf, 
        attn_drop_prob=.0, 
        lin_drop_prob=.0, 
        freeze_12=freeze_12,
        device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    train(
        train_loader, 
        swin_type, 
        dataset, 
        epochs, 
        model, 
        lf, 
        ltoken_num,
        optimizer, 
        criterion, 
        device, 
        show_per=show_per,
        reg_type=reg_type, 
        reg_lambda=reg_lambda, 
        validation=val_loader)