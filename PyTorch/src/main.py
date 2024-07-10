import torch
import torch.nn as nn
import torch.optim as optim
from dataset import FSD50K, ESC50
from mergenet import MergeNet
from effnet import EfficientNet
from train_eval_funcs import *

# DEMO
def execute_demo(dataset, model, plain_mergenet, pretrained, imagenet_pretrain, to_train, seed):
    if seed:
        fix_seed()
    train_data, val_data, eval_data = load_dataset(dataset)
    model, criterion, optimizer, device = load_model(model, dataset, pretrained, to_train, plain_mergenet=plain_mergenet, imagenet_pretrain=imagenet_pretrain)
    if dataset == "FSD50K":
        if to_train:
            train(model, train_data, BATCH_SIZE, criterion, optimizer, NUM_EPOCHS, device, val_data=val_data)
        eval(model, eval_data, BATCH_SIZE, criterion, device, type="Test")
    else:
        epochs_per_fold = EPOCHS_PER_FOLD_FINETUNING if pretrained else EPOCHS_PER_FOLD_TRAINING
        cross_val(model, train_data, train_data.get_fold_indices(), BATCH_SIZE, criterion, optimizer, epochs_per_fold, device)


#########################################################################################################

SEED = 1234
LR = 1e-3
BATCH_SIZE = 8
NUM_EPOCHS = 30 # FSD50K
EPOCHS_PER_FOLD_FINETUNING = 5 # ESC50
EPOCHS_PER_FOLD_TRAINING = 10 # ESC50

#########################################################################################################

def fix_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#########################################################################################################

def load_dataset(dataset):
    assert dataset in ["FSD50K", "ESC50"]
    
    if dataset == "FSD50K":
        train_data = FSD50K(fsd50k_dev_audio_path, fsd50k_dev_spectro_path, fsd50k_gt_path, split="train")
        val_data = FSD50K(fsd50k_dev_audio_path, fsd50k_dev_spectro_path, fsd50k_gt_path, split="val")
        eval_data = FSD50K(fsd50k_eval_audio_path, fsd50k_eval_spectro_path, fsd50k_gt_path, split="test")
    else:
        train_data = ESC50(esc50_audio_path, esc50_spectro_path, esc50_gt_path)
        val_data = []
        eval_data = []

    return train_data, val_data, eval_data

#########################################################################################################

def load_model(model_name, dataset, pretrained, to_train, plain_mergenet=False, imagenet_pretrain=False):
    assert model_name in ["MergeNet","EfficientNet"]

    label_size = {"FSD50K":200, "ESC50":50}
    finetune = {"FSD50K":False, "ESC50":True}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == "MergeNet":
        model = MergeNet(num_classes=label_size[dataset], split=not plain_mergenet)
    else:
        model = EfficientNet(num_classes=label_size[dataset], pretrained=imagenet_pretrain)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if (dataset == "ESC50" and pretrained) or (dataset == "FSD50K" and not to_train):
        load_checkpoint(model, optimizer=optimizer, device=device, finetuning=finetune[dataset])

    model = model.to(device)

    return model, criterion, optimizer, device

#########################################################################################################


if __name__ == "__main__":
    pass