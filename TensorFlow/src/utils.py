import os
import csv
from math import floor
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

# In MyDataLoader "self.indices" is used to know the exact order of data samples after they're shuffled by the DataLoader itself;
# the order is, in fact, needed when the MixUp technique is used because online preprocessing is implemented with a queue
# so that, to avoid wasting time, samples having indices of those that have to be mixed are preprocessed as soon as possible
# and inserted in the queue, this is done in background by another Thread and totally on CPU in the meantime that samples not
# to be mixed (whose audio features, already preprocessed, just need to be loaded from disk) are sent for processing to the GPU.
# In other words, online preprocessing is not on the fly but anticipated in time before mixup-chosen audios are requested by the Dataloader, see the report for more.
class MyDataLoader(tf.keras.utils.Sequence):
    def __init__(self, data, indices, batch_size):
        self.data = data
        self.len = floor(len(indices) / batch_size)
        self.batch_size = batch_size
        self.indices = indices
    
    def reset_indices(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_features, batch_labels = [], []
        for idx in range(start, end):
            features, labels = self.data[self.indices[idx]]
            batch_features.append(features)
            batch_labels.append(labels)
        return tf.convert_to_tensor(batch_features), tf.convert_to_tensor(batch_labels)

class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def call(self, x):
        batch_size, height, width, channels = tf.unstack(tf.shape(x))
        x_reshaped = tf.reshape(x, [batch_size, height, width, -1, self.groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        x_shuffled = tf.reshape(x_transposed, [batch_size, height, width, channels])
        return x_shuffled

def compute_mAP(preds, labels):
    APs = []
    for label in range(labels.shape[1]):
        AP = average_precision_score(labels[:, label], preds[:, label])
        APs.append(AP)
    mAP = np.mean(APs)
    return mAP

class CSVLogger:
    def __init__(self, csv_name):
        self.csv_path = "logs/"+csv_name+".csv"
        self.fieldnames = ['epoch', 'training_loss', 'validation_loss']

        if not os.path.exists(self.csv_path):
            os.makedirs("logs", exist_ok=True)
            with open(self.csv_path, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, epoch, training_loss, validation_loss):
        with open(self.csv_path, 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writerow({'epoch': epoch, 'training_loss': training_loss, 'validation_loss': validation_loss})


#################################################FSD50K#################################################

fsd50k_dev_audio_path = "/home/luca/data/FSD50K/FSD50K.dev_audio_16KHz_10s"
fsd50k_eval_audio_path = "/home/luca/data/FSD50K/FSD50K.eval_audio_16KHz_10s"

fsd50k_dev_spectro_path = "/home/luca/data/FSD50K/FSD50K.dev_audio_spectro"
fsd50k_eval_spectro_path = "/home/luca/data/FSD50K/FSD50K.eval_audio_spectro"

fsd50k_gt_path = "/home/luca/data/FSD50K/FSD50K.ground_truth"

#################################################ESC50#################################################

esc50_audio_path = "/home/luca/data/ESC50/16000"

esc50_spectro_path = "/home/luca/data/ESC50/spectros"

esc50_gt_path = "/home/luca/data/ESC50"

#######################################################################################################