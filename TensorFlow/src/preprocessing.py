import os
import glob
import shutil
import argparse
from multiprocessing import Pool
import numpy as np
from math import ceil
import librosa

def extract_features(y, sr):
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
    y_centered = y_resampled - y_resampled.mean()
    
    n_fft = int(0.025 * 16000)  # 25ms window
    hop_length = int(0.010 * 16000)  # 10ms hop
    mel_features = librosa.feature.melspectrogram(y=y_centered, sr=16000, n_fft=n_fft, 
                                                  hop_length=hop_length, n_mels=128, window="hann")
    mel_features_db = librosa.power_to_db(mel_features)
    
    mean = mel_features_db.mean()
    std = mel_features_db.std()
    mel_features_norm = (mel_features_db - mean) / (std+1e-7)
    
    return mel_features_norm

def mixup(mixup_audios, mixup_labels, audios, labels, N, MixUpQueue):
    for sample, label in zip(mixup_audios, mixup_labels):
        ids = np.random.choice(range(len(audios)), N, replace=False)
        files = [sample]+[audios[i] for i in ids]
        wfs = [librosa.load(f, sr=None, duration=10.0)[0] for f in files]
        wfs = [w-w.mean() for w in wfs]
        max_len = max([len(w) for w in wfs])
        wfs = [np.tile(w, ceil(max_len/len(w)))[:max_len] for w in wfs]
        y = np.mean(wfs, axis=0)
        mixed_features = extract_features(y, 16000)
        mixed_labels = np.mean([label]+[labels[i] for i in ids], axis=0)

        MixUpQueue.put(mixed_features)
        MixUpQueue.put(mixed_labels)

def process_idx(idx):
    f = files[idx]
    y, sr = librosa.load(f, sr=None, duration=10.0)
    features = extract_features(y, sr)
    
    fname = f.split("/")[-1].split(".")[0]
    np.save('{}.npy'.format(os.path.join(tgt_dir, fname)), features)

    if idx % 1000 == 0:
        print(f"Done: {idx}/{lf}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = "Multiprocessing script to create spectrograms from raw audios"
    parser.add_argument("--src_path", "-s", type=str,
                        help="source directory containing .wav files")
    parser.add_argument("--dst_path", "-t", type=str,
                        help="target directory where .npy files will be stored")
    args = parser.parse_args()
    
    files = glob.glob("{}/*.wav".format(args.src_path))
    lf = len(files)
    tgt_dir = args.dst_path

    if os.path.exists(tgt_dir):
        shutil.rmtree(tgt_dir)
    os.makedirs(tgt_dir)

    with Pool(16) as pool:
        pool.map(process_idx, range(lf))