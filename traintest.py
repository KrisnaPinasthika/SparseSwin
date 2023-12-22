import torch
import os 
import numpy as np 
import pandas as pd

def train(train_loader, swin_type, dataset, epochs, model, lf, ltoken_num,
            optimizer, criterion, device, show_per,  
            reg_type=None, reg_lambda=0., validation=None):
    model.train()
    total_batch = train_loader.__len__()
    train_test_hist = []
    best_test_acc = -99
    
    specific_dir = f'./SavedModel/{dataset}/SparseSwin_reg_{reg_type}_lbd_{reg_lambda}_lf_{lf}_{ltoken_num}'
    if f'SparseSwin_reg_{reg_type}_lbd_{reg_lambda}_lf_{lf}_{ltoken_num}' not in os.listdir(f'./SavedModel/{dataset}/'): 
        os.mkdir(specific_dir)
    
    print(f"[TRAIN] Total : {total_batch} | type : {swin_type} | Regularization : {reg_type} with lamda : {reg_lambda}")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        running_loss, n_correct, n_sample = 0.0, 0.0, 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, attn_weights = model(inputs)
            
            reg = 0
            if reg_type == 'l1':                
                for attn_w in attn_weights: 
                    reg += torch.sum(torch.abs(attn_w))
                    
            elif reg_type == 'l2':
                for attn_w in attn_weights: 
                    reg += torch.sum(attn_w**2)
                        
            reg = reg_lambda * reg
            
            loss = criterion(outputs, labels) + reg
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad(): 
                n_correct_per_batch = torch.sum(torch.argmax(outputs, dim=1) == labels)
                n_correct += n_correct_per_batch
                n_sample += labels.shape[0]
                acc = n_correct / n_sample

            if ((i + 1) % show_per == 0) or ((i + 1) == total_batch):
                print(f'  [{i + 1}/{total_batch}] Loss: {(running_loss / (i + 1)):.4f} Acc : {acc:.4f}')

        print(f'Loss: {(running_loss / total_batch):.4f} Acc : {(n_correct / n_sample):.4f}')
        
        # Save model
        test_loss, test_acc = test(validation, swin_type=swin_type, model=model, criterion=criterion, device=device)
        train_loss, train_acc = (running_loss / total_batch), (n_correct / n_sample)

        test_loss, train_loss = round(test_loss, 4), round(train_loss, 4)
        train_test_hist.append([train_loss, round(train_acc.item(), 4), test_loss, round(test_acc.item(), 4)])
        
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'{specific_dir}/model_{epoch+1}.pt')
            print(f"--> Best Model saved successfully | Acc : {best_test_acc:.4f}")
    
    train_test_hist = np.array(train_test_hist)
    df = pd.DataFrame()
    df['train_loss'] = train_test_hist[:, 0]
    df['train_acc'] = train_test_hist[:, 1]
    df['test_loss'] = train_test_hist[:, 2]
    df['test_acc'] = train_test_hist[:, 3]
    df.to_csv(f'{specific_dir}/hist.csv', index=None)

def test(val_loader, swin_type, model, criterion, device):
    model.eval()

    with torch.no_grad():
        total_batch = val_loader.__len__()
        print(f"[TEST] Total : {total_batch} | type : {swin_type}")
        running_loss, n_correct, n_sample = 0.0, 0.0, 0.0

        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward 
            outputs, attn_weights = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            n_correct_per_batch = torch.sum(torch.argmax(outputs, dim=1) == labels)
            n_correct += n_correct_per_batch
            n_sample += labels.shape[0]

    print(f'[Model : {swin_type}] Loss: {(running_loss / total_batch):.4f} Acc : {(n_correct / n_sample):.4f}')
    print()
    return (running_loss / total_batch), (n_correct / n_sample)