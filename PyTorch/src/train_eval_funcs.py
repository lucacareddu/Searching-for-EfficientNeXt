import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *

# for ESC50 only
def cross_val(model, data, folds, batch_size, criterion, optimizer, epochs_per_fold, device, num_folds=5, mixup=False):
    accuracies = []

    for fold in range(1,num_folds+1):
        print(f"Fold {fold}:")

        train_indices = np.where(folds != fold)[0]
        test_indices = np.where(folds == fold)[0]

        train(model, data, batch_size, criterion, optimizer, epochs_per_fold, device, indices=train_indices, mixup=mixup)
        accuracies.append(eval(model, data, batch_size, criterion, device, indices=test_indices)[1])
    
    avg_accuracy = np.mean(accuracies)
    print(f"Estimated Test Accuracy: {avg_accuracy:.4f}")


def train(model, train_data, batch_size, criterion, optimizer, num_epochs, device, val_data=None, indices=None, mixup=True):
    assert (val_data is None) ^ (indices is None) # FSD50K (val) or ESC50 (cross-val)
    fsd50k = indices is None

    if fsd50k:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'models/{timestamp}/'
        os.makedirs(save_dir)

        log = CSVLogger(csv_name=timestamp)

    rsampler = RSampler(list(range(len(train_data)))) if fsd50k else RSampler(indices)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=rsampler)

    if fsd50k:
        best_mAP = 0.0

    for epoch in range(num_epochs):
        model.train()
        tot_loss = 0.0

        rsampler.reset_indices()
        if mixup:
            train_data.reset_mixup(rsampler.indices)

        loop = tqdm(train_dataloader, total=len(train_dataloader), leave=True)

        for data, labels in loop:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            tot_loss += loss.item()
        
        train_loss = tot_loss/len(train_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}")

        if not fsd50k:
            continue

        ##############
        # Validation #
        ##############

        val_loss, mAP_score = eval(model, val_data, batch_size, criterion, device)

        if mAP_score > best_mAP:
            filename = f'best_{epoch+1}.pth'
            best_mAP = mAP_score
        else:
            filename = f'model_{epoch+1}.pth'

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, filename))

        log.log(epoch+1, train_loss, val_loss)


def eval(model, eval_data, batch_size, criterion, device, indices=None, type="Validation"):
    fsd50k = indices is None
    sampler = None if fsd50k else RSampler(indices)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, sampler=sampler)

    model.eval()
    tot_loss = 0.0

    if fsd50k: # mAP
        all_preds = []
        all_labels = []
    else: # Accuracy
        corr = 0
        tot = 0
    
    loop = tqdm(eval_dataloader, total=len(eval_dataloader), leave=False)

    with torch.no_grad():
        for data, labels in loop:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            tot_loss += loss.item()

            if fsd50k:
                all_preds.append(outputs.cpu().detach())
                all_labels.append(labels.cpu().detach())
            else:
                corr += np.sum(np.argmax(labels.cpu().detach().numpy(), axis=-1) == np.argmax(outputs.cpu().detach().numpy(), axis=-1))
                tot += batch_size

    eval_loss = tot_loss/len(eval_dataloader)
    
    if fsd50k:
        mAP_score = compute_mAP(np.vstack(all_preds), np.vstack(all_labels))
        print(f"{type} Loss: {eval_loss}, mAP: {mAP_score}")
        return eval_loss, mAP_score
    else:
        accuracy = corr/tot
        print(f"{type} Loss: {eval_loss}, Accuracy: {accuracy}")
        return eval_loss, accuracy