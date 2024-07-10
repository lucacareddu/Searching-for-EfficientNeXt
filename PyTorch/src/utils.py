import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Sampler
from sklearn.metrics import average_precision_score

# RSampler is used to know the exact order of data samples after they're shuffled by the DataLoader;
# the order is, in fact, needed when the MixUp technique is used because online preprocessing is implemented with a queue
# so that, to avoid wasting time, samples having indices of those that have to be mixed are preprocessed as soon as possible
# and inserted in the queue, this is done in background by another Thread and totally on CPU in the meantime that samples not
# to be mixed (whose audio features, already preprocessed, just need to be loaded from disk) are sent for processing to the GPU.
# In other words, online preprocessing is not on the fly but anticipated in time before mixup-chosen audios are requested by the Dataloader, see the report for more.
class RSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)
    def reset_indices(self):
        np.random.shuffle(self.indices)

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x

def compute_mAP(preds, labels):
    APs = []
    for label in range(labels.shape[1]):
        AP = average_precision_score(labels[:, label], preds[:, label])
        APs.append(AP)
    mAP = np.mean(APs)
    return mAP

def load_checkpoint(model, optimizer=None, device=None, finetuning=False):
    assert not ((optimizer is None) ^ (device is None))

    model_name = model.__class__.__name__
    if model_name == "MergeNet":
        path = fsd50k_mergenet_weights if model.split else fsd50k_plain_mergenet_weights
    else:
        path = fsd50k_pretrained_efficientnetb2_weights if model.pretrained else fsd50k_efficientnetb2_weights
    checkpoint = torch.load(path)

    if finetuning:
        trainables = {k:v for k,v in checkpoint['model_state_dict'].items() if 'fc' not in k}
        model.load_state_dict(trainables, strict=False)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

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

fsd50k_mergenet_weights = "Experiments/MergeNet/MergeNet_32Split/best_27.pth"
                        # with channels shuffling is "Experiments/MergeNet/MergeNet_Shuffle/best_27.pth"
                        # with mnblock_sep is "Experiments/MergeNet/MergeNet_32Split_Sep/model_30.pth"
                        # with 16_split is "Experiments/MergeNet/MergeNet_16Split/best_25.pth"
                        # with 64_split is "Experiments/MergeNet/MergeNet_64Split/best_25.pth"
fsd50k_plain_mergenet_weights = "Experiments/MergeNet/MergeNet_NoSplit/model_28.pth"
fsd50k_efficientnetb2_weights = "Experiments/EfficientNet/EffNetB2/best_27.pth"
                        # with tf_efficientnetv2_b2 is "Experiments/EfficientNet/EffNetV2B2/best_24.pth"
fsd50k_pretrained_efficientnetb2_weights = "Experiments/EfficientNet/EffNetB2_ImageNet/model_30.pth"
                        # with tf_efficientnetv2_b2 is "Experiments/EfficientNet/EffNetV2B2_ImageNet/best_22.pth"

#################################################ESC50#################################################

esc50_audio_path = "/home/luca/data/ESC50/16000"

esc50_spectro_path = "/home/luca/data/ESC50/spectros"

esc50_gt_path = "/home/luca/data/ESC50"

#######################################################################################################