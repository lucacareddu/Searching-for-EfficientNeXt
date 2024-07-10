import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils import *

# for ESC50 only
def cross_val(model, data, folds, batch_size, criterion, optimizer, epochs_per_fold, num_folds=5, mixup=False):
    accuracies = []

    for fold in range(1,num_folds+1):
        print(f"Fold {fold}:")

        train_indices = np.where(folds != fold)[0]
        test_indices = np.where(folds == fold)[0]

        train(model, data, batch_size, criterion, optimizer, epochs_per_fold, indices=train_indices, mixup=mixup)
        accuracies.append(eval(model, data, batch_size, criterion, indices=test_indices)[1])
    
    avg_accuracy = np.mean(accuracies)
    print(f"Estimated Test Accuracy: {avg_accuracy:.4f}")


def train(model, train_data, batch_size, criterion, optimizer, num_epochs, val_data=None, indices=None, mixup=True):
    assert (val_data is None) ^ (indices is None) # FSD50K (val) or ESC50 (cross-val)
    fsd50k = indices is None

    if fsd50k:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'models/{timestamp}/'
        os.makedirs(save_dir)

        log = CSVLogger(csv_name=timestamp)
    
    train_dataloader = MyDataLoader(train_data, list(range(len(train_data))), batch_size) if fsd50k else MyDataLoader(train_data, indices, batch_size)
    
    if fsd50k:
        best_mAP = 0.0

    for epoch in range(num_epochs):
        tot_loss = 0.0

        train_dataloader.reset_indices()
        if mixup:
            train_data.reset_mixup(train_dataloader.indices)

        loop = tqdm(train_dataloader, total=len(train_dataloader), leave=True)

        for data, labels in loop:
                
            with tf.GradientTape() as tape:
                outputs = model(data, training=True)
                loss = criterion(labels, outputs)
                gradients = tape.gradient(loss, model.trainable_variables)

                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            tot_loss += loss.numpy()
        
        train_loss = tot_loss/len(train_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}")

        if not fsd50k:
            continue

        ##############
        # Validation #
        ##############

        val_loss, mAP_score = eval(model, val_data, batch_size, criterion)

        if mAP_score > best_mAP:
            filename = f'best_{epoch+1}'
            best_mAP = mAP_score
        else:
            filename = f'model_{epoch+1}'

        model.save(os.path.join(save_dir, filename), save_format='tf')

        log.log(epoch+1, train_loss, val_loss)


def eval(model, eval_data, batch_size, criterion, indices=None, type="Validation"):
    fsd50k = indices is None
    eval_dataloader = MyDataLoader(eval_data, list(range(len(eval_data))), batch_size) if fsd50k else MyDataLoader(eval_data, indices, batch_size)

    tot_loss = 0.0

    if fsd50k: # mAP
        all_preds = []
        all_labels = []
    else: # Accuracy
        corr = 0
        tot = 0
    
    loop = tqdm(eval_dataloader, total=len(eval_dataloader), leave=False)

    for data, labels in loop:
        outputs = model(data, training=False)
        loss = criterion(outputs, labels)
        tot_loss += loss.numpy()

        if fsd50k:
            all_preds.append(outputs.numpy())
            all_labels.append(labels.numpy())
        else:
            corr += tf.reduce_sum(tf.cast(tf.argmax(labels, axis=-1) == tf.argmax(outputs, axis=-1), tf.int32))
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