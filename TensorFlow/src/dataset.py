import os
import numpy as np
from math import ceil
import pandas as pd
import tensorflow as tf
from preprocessing import mixup
import threading
import queue

class FSD50K(tf.keras.utils.Sequence):
    def __init__(self, audios_path, spectros_path, gt_path, split):
        super(FSD50K, self).__init__()

        assert split in ["train", "val", "test"]
        self.split = split

        voc_csv = pd.read_csv(os.path.join(gt_path,"vocabulary.csv"), sep=",", names = ['index', 'fname', 'mids'])
        self.labels_map = dict(zip(voc_csv['fname'], voc_csv['index']))

        if split in ["train", "val"]:
            dev_csv = pd.read_csv(os.path.join(gt_path,"dev.csv"), sep=",")
            offset = dev_csv[dev_csv.split == split]
            self.audios = offset.fname.apply(lambda n: os.path.join(audios_path, f"{n}.wav")).tolist()
            self.spectros = offset.fname.apply(lambda n: os.path.join(spectros_path, f"{n}.npy")).tolist()
            self.labels = offset.labels.apply(self.convert_labels).tolist()
            if split == "train":
                self.mixup_indices = []
                self.mixup_queue = queue.Queue()
        else:
            eval_csv = pd.read_csv(os.path.join(gt_path,"eval.csv"), sep=",")
            self.audios = eval_csv.fname.apply(lambda n: os.path.join(audios_path, f"{n}.wav")).tolist()
            self.spectros = eval_csv.fname.apply(lambda n: os.path.join(spectros_path, f"{n}.npy")).tolist()
            self.labels = eval_csv.labels.apply(self.convert_labels).tolist()
        
        self.len = len(self.audios)
    
    def convert_labels(self, label_str):
        str_labels = label_str.split(',')
        int_labels = [self.labels_map[label] for label in str_labels]
        bin_labels = [1 if i in int_labels else 0 for i in range(200)]
        return bin_labels
    
    def features_getter(self, idx=None, features=None):
        assert (idx is None) ^ (features is None)
        features = features if features is not None else np.load(self.spectros[idx])
        if features.shape[1] < 1000:
            features = np.tile(features, (1, ceil(1000/features.shape[1])))
        return features[:,:1000]
        
    def reset_mixup(self, shuffled_indices, mixup_divisor=2, mixup_num=1):
        assert self.split == "train"
        rand_indices = np.random.choice(range(self.len), self.len//mixup_divisor, replace=False)
        self.mixup_indices = [shuffled_indices[i] for i in range(len(shuffled_indices)) if i in rand_indices] # order is relevant
        mixup_audios = [self.audios[i] for i in self.mixup_indices]
        mixup_labels = [self.labels[i] for i in self.mixup_indices]
        
        args=[mixup_audios, mixup_labels, self.audios, self.labels, mixup_num, self.mixup_queue]
        threading.Thread(target=mixup, args=args).start()

    def __getitem__(self, idx):
        if self.split == "train" and idx in self.mixup_indices: # MixUp
            features = tf.expand_dims(tf.convert_to_tensor(self.features_getter(features=self.mixup_queue.get())), 2)
            labels = tf.convert_to_tensor(self.mixup_queue.get(), dtype=tf.float32)
        else:
            features = tf.expand_dims(tf.convert_to_tensor(self.features_getter(idx=idx)), 2)
            labels = tf.convert_to_tensor(self.labels[idx], dtype=tf.float32)
        return features, labels

    def __len__(self):
        return self.len
    

class ESC50(tf.keras.utils.Sequence):
    def __init__(self, audios_path, spectros_path, gt_path):
        super(ESC50, self).__init__()
        
        csv = pd.read_csv(os.path.join(gt_path,"esc50.csv"), sep=",")
        self.audios = csv.filename.apply(lambda n: os.path.join(audios_path, n)).tolist()
        self.spectros = csv.filename.apply(lambda n: os.path.join(spectros_path, f"{n[:-4]}.npy")).tolist()
        self.labels = csv.target.apply(self.convert_labels).tolist()
        self.fold_indices = csv.fold.values
        self.len = len(self.audios)

        self.mixup_indices = []
        self.mixup_queue = queue.Queue()

    def convert_labels(self, label_str):
        bin_label = [1 if i==int(label_str) else 0 for i in range(50)]
        return bin_label
    
    def get_fold_indices(self):
        return self.fold_indices
    
    def features_getter(self, idx=None, features=None):
        assert (idx is None) ^ (features is None)
        features = features if features is not None else np.load(self.spectros[idx])
        if features.shape[1] < 1000:
            features = np.tile(features, (1, ceil(1000/features.shape[1])))
        return features[:,:1000]

    def reset_mixup(self, shuffled_indices, mixup_divisor=2, mixup_num=1):
        rand_indices = np.random.choice(range(len(shuffled_indices)), len(shuffled_indices)//mixup_divisor, replace=False)
        self.mixup_indices = [shuffled_indices[i] for i in range(len(shuffled_indices)) if i in rand_indices] # order is relevant
        mixup_audios = [self.audios[i] for i in self.mixup_indices]
        mixup_labels = [self.labels[i] for i in self.mixup_indices]

        allowed_audios = [self.audios[i] for i in shuffled_indices] #depend on the fold
        allowed_labels = [self.labels[i] for i in shuffled_indices] #depend on the fold
        
        args=[mixup_audios, mixup_labels, allowed_audios, allowed_labels, mixup_num, self.mixup_queue]
        threading.Thread(target=mixup, args=args).start()
    
    def __getitem__(self, idx):
        if idx in self.mixup_indices: # MixUp
            features = tf.expand_dims(tf.convert_to_tensor(self.features_getter(features=self.mixup_queue.get())), 2)
            labels = tf.convert_to_tensor(self.mixup_queue.get(), dtype=tf.float32)
        else:
            features = tf.expand_dims(tf.convert_to_tensor(self.features_getter(idx=idx)), 2)
            labels = tf.convert_to_tensor(self.labels[idx], dtype=tf.float32)
        return features, labels

    def __len__(self):
        return self.len