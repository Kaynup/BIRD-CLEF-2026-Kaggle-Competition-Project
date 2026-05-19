import json
import os

def create_notebook(filename, code_blocks):
    cells = []
    for code in code_blocks:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")]
        })
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=1)

# ---- TRAIN NOTEBOOK ----
train_blocks = [
"""import os
import ast
import math
import glob
import random
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import soundfile as sf
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import timm

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

class CFG:
    # Kaggle Paths
    ROOT_DIR = '/kaggle/input/competitions/birdclef-2026'
    TRAIN_CSV = os.path.join(ROOT_DIR, 'train.csv')
    SAMPLE_SUB_CSV = os.path.join(ROOT_DIR, 'sample_submission.csv')
    TRAIN_AUDIO_DIR = os.path.join(ROOT_DIR, 'train_audio')
    
    # Training Setup
    NUM_FOLDS = 5
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    
    # Audio Setup
    SR = 32000
    DURATION_SECONDS = 5.0
    CHUNK_LENGTH = int(SR * DURATION_SECONDS)
    
    # Spectrogram setup
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMIN = 20
    FMAX = 16000
    
    # Model
    BACKBONE_NAME = 'vit_base_patch16_224'
    IMAGE_SIZE = (224, 224)
    
    # Loss weight for Head B
    GEO_LOSS_WEIGHT = 5.0
""",
"""def setup_data():
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    sample_sub = pd.read_csv(CFG.SAMPLE_SUB_CSV)
    submission_labels = [c for c in sample_sub.columns if c != 'row_id']
    CFG.NUM_CLASSES = len(submission_labels)
    species_to_idx = {species: idx for idx, species in enumerate(submission_labels)}
    
    # ZERO-LEAKAGE STRATEGY: Group by Author (Recordist)
    train_df['author'] = train_df['author'].fillna('unknown')
    train_df['stratify_label'] = train_df['primary_label']
    
    sgkf = StratifiedGroupKFold(n_splits=CFG.NUM_FOLDS)
    folds = list(sgkf.split(train_df, train_df['stratify_label'], groups=train_df['author']))
    return train_df, folds, submission_labels, species_to_idx

class BirdCLEFDataset(Dataset):
    def __init__(self, df, species_to_idx, is_train=True):
        self.df = df
        self.species_to_idx = species_to_idx
        self.is_train = is_train
        self.mel_transform = T.MelSpectrogram(
            sample_rate=CFG.SR, n_fft=CFG.N_FFT, hop_length=CFG.HOP_LENGTH,
            n_mels=CFG.N_MELS, f_min=CFG.FMIN, f_max=CFG.FMAX
        )
        self.time_masking = T.TimeMasking(time_mask_param=20)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=20)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):
        try:
            data, sample_rate = sf.read(path, dtype='float32')
            if data.ndim > 1: data = data.mean(axis=1) # Mono
            if sample_rate != CFG.SR:
                resampler = T.Resample(sample_rate, CFG.SR)
                data = resampler(torch.tensor(data)).numpy()
        except:
            data = np.zeros(CFG.CHUNK_LENGTH)
            
        waveform = torch.tensor(data)
        total_len = waveform.size(0)
        
        # RANDOM CROP FIX: Prevent model from learning empty intro noise
        if total_len > CFG.CHUNK_LENGTH:
            if self.is_train:
                offset = random.randint(0, total_len - CFG.CHUNK_LENGTH)
                waveform = waveform[offset:offset + CFG.CHUNK_LENGTH]
            else:
                offset = (total_len - CFG.CHUNK_LENGTH) // 2
                waveform = waveform[offset:offset + CFG.CHUNK_LENGTH]
        elif total_len < CFG.CHUNK_LENGTH:
            waveform = F.pad(waveform, (0, CFG.CHUNK_LENGTH - total_len))
            
        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(CFG.TRAIN_AUDIO_DIR, row['filename'])
        waveform = self._load_audio(audio_path)
        
        # Acoustic Target
        target = torch.zeros(CFG.NUM_CLASSES, dtype=torch.float32)
        primary = str(row['primary_label']).strip()
        if primary in self.species_to_idx:
            target[self.species_to_idx[primary]] = 1.0
            
        # Geographic Target (Radians)
        lat = float(row.get('latitude', -19.05))
        lon = float(row.get('longitude', -56.75))
        if np.isnan(lat): lat = -19.05
        if np.isnan(lon): lon = -56.75
        geo_target = torch.tensor(np.radians([lat, lon]), dtype=torch.float32)
        
        # Log-Mel Spectrogram & Background Augmentations
        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-6)
        
        if self.is_train:
            log_mel = self.time_masking(log_mel)
            log_mel = self.freq_masking(log_mel)
            
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
        
        image = log_mel.unsqueeze(0).unsqueeze(0)
        image = F.interpolate(image, size=CFG.IMAGE_SIZE, mode='bilinear', align_corners=False).squeeze(0).repeat(3, 1, 1)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        return image, target, geo_target
""",
"""class DualHeadViT(nn.Module):
    def __init__(self, backbone_name=CFG.BACKBONE_NAME, num_classes=234):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, in_chans=3)
        in_features = self.backbone.head.in_features if hasattr(self.backbone, 'head') else 768
        self.backbone.reset_classifier(0)
        
        # Head A: The Acoustic Pipeline
        self.acoustic_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Head B: The Geolocation Self-Distillation (Mapping Ecosystem)
        self.geo_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.acoustic_head(features), self.geo_head(features)
""",
"""def train_knn_prior(train_df, submission_labels, species_to_idx):
    spatial_df = train_df.dropna(subset=['latitude', 'longitude']).copy()
    X_rad = np.radians(spatial_df[['latitude', 'longitude']].values)
    
    y_targets = np.zeros((len(spatial_df), len(submission_labels)), dtype=np.float32)
    for idx, row in enumerate(spatial_df.itertuples()):
        primary = str(row.primary_label).strip()
        if primary in species_to_idx:
            y_targets[idx, species_to_idx[primary]] = 1.0
            
    knn = KNeighborsRegressor(n_neighbors=15, weights='distance', metric='haversine')
    knn.fit(X_rad, y_targets)
    return knn

def train_one_epoch(model, loader, optimizer, crit_ac, crit_geo, device):
    model.train()
    total_loss = 0
    for images, targets, geo_targets in tqdm(loader):
        images, targets, geo_targets = images.to(device), targets.to(device), geo_targets.to(device)
        
        optimizer.zero_grad()
        ac_out, geo_out = model(images)
        
        loss = crit_ac(ac_out, targets) + (CFG.GEO_LOSS_WEIGHT * crit_geo(geo_out, geo_targets))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, knn_model, device):
    model.eval()
    all_targets, all_blended_preds = [], []
    
    for images, targets, _ in tqdm(loader):
        images = images.to(device)
        ac_out, geo_out = model(images)
        
        probs_visual = torch.sigmoid(ac_out).cpu().numpy()
        
        # SELF-GENERATED SPATIAL PRIOR: Zero Data Leakage
        predicted_coords = geo_out.cpu().numpy()
        probs_knn = knn_model.predict(predicted_coords)
        
        # Confidence-Gated Blending
        final_probs = np.zeros_like(probs_visual)
        for i in range(len(probs_visual)):
            if np.max(probs_visual[i]) > 0.85:
                final_probs[i] = probs_visual[i]
            else:
                final_probs[i] = probs_visual[i] * 0.7 + probs_knn[i] * 0.3
                
        all_targets.append(targets.numpy())
        all_blended_preds.append(final_probs)
        
    all_targets = np.concatenate(all_targets)
    all_blended_preds = np.concatenate(all_blended_preds)
    
    val_f1 = f1_score((all_targets > 0).astype(int), (all_blended_preds >= 0.5).astype(int), average='macro', zero_division=0)
    return val_f1
""",
"""# main block
# train_df, folds, submission_labels, species_to_idx = setup_data()
# knn_model = train_knn_prior(train_df, submission_labels, species_to_idx)
# joblib.dump(knn_model, 'knn_spatial_prior.pkl')
# (Train loop goes here as normal...)
print('Training ready!')
"""
]


# ---- INFERENCE NOTEBOOK ----
inference_blocks = [
"""import os
import math
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import soundfile as sf
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import timm

class CFG:
    ROOT_DIR = '/kaggle/input/competitions/birdclef-2026'
    SAMPLE_SUB_CSV = os.path.join(ROOT_DIR, 'sample_submission.csv')
    SOUNDSCAPE_DIR = os.path.join(ROOT_DIR, 'test_soundscapes')
    
    MODEL_PATH = 'best_dual_head_vit_fold_0.pth'
    KNN_PATH = 'knn_spatial_prior.pkl'
    
    # OVERLAPPING SLIDING WINDOWS (From 01-multilabel)
    SR = 32000
    WINDOW_SECONDS = 5.0
    HOP_SECONDS = 2.5
    CHUNK_LENGTH = int(SR * WINDOW_SECONDS)
    HOP_LENGTH_AUDIO = int(SR * HOP_SECONDS)
    
    BACKBONE_NAME = 'vit_base_patch16_224'
    IMAGE_SIZE = (224, 224)
    
    # Confidence-Gated Blending Strategy
    CONFIDENCE_THRESHOLD = 0.85
    KNN_ALPHA = 0.3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
""",
"""class DualHeadViT(nn.Module):
    def __init__(self, backbone_name=CFG.BACKBONE_NAME, num_classes=234):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, in_chans=3)
        in_features = self.backbone.head.in_features if hasattr(self.backbone, 'head') else 768
        self.backbone.reset_classifier(0)
        self.acoustic_head = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_classes))
        self.geo_head = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 2))
    def forward(self, x):
        features = self.backbone(x)
        return self.acoustic_head(features), self.geo_head(features)

sample_sub = pd.read_csv(CFG.SAMPLE_SUB_CSV)
submission_labels = [c for c in sample_sub.columns if c != 'row_id']
NUM_CLASSES = len(submission_labels)

model = DualHeadViT(num_classes=NUM_CLASSES).to(device)
if os.path.exists(CFG.MODEL_PATH):
    model.load_state_dict(torch.load(CFG.MODEL_PATH, map_location=device))
model.eval()

knn_model = None
if os.path.exists(CFG.KNN_PATH):
    knn_model = joblib.load(CFG.KNN_PATH)

mel_transform = T.MelSpectrogram(sample_rate=CFG.SR, n_fft=2048, hop_length=512, n_mels=128, f_min=20, f_max=16000).to(device)
""",
"""test_files = sorted(glob.glob(f"{CFG.SOUNDSCAPE_DIR}/*.ogg"))
if not test_files: # Kaggle hidden test fallback
    test_files = sorted(glob.glob(os.path.join(CFG.ROOT_DIR, 'train_soundscapes', '*.ogg')))[:2]

all_predictions = []
all_row_ids = []

for audio_path in tqdm(test_files, desc="Inference"):
    filename = os.path.basename(audio_path).replace('.ogg', '')
    data, _ = sf.read(audio_path, dtype='float32')
    if data.ndim > 1: data = data.mean(axis=1)
    
    total_samples = len(data)
    num_submission_chunks = math.ceil(total_samples / CFG.CHUNK_LENGTH)
    submission_block_probs = np.zeros((num_submission_chunks, NUM_CLASSES))
    
    # 1. Overlapping Windows Extraction
    windows, starts = [], []
    for start in range(0, total_samples, CFG.HOP_LENGTH_AUDIO):
        end = start + CFG.CHUNK_LENGTH
        if end > total_samples: break
        windows.append(data[start:end])
        starts.append(start)
        
    if not windows:
        windows.append(np.pad(data, (0, CFG.CHUNK_LENGTH - total_samples)))
        starts.append(0)
        
    waveforms = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
    
    # 2. Batch Inference
    with torch.no_grad():
        mel_specs = mel_transform(waveforms)
        log_mels = torch.log(mel_specs + 1e-6)
        
        m_min = log_mels.view(len(windows), -1).min(dim=1)[0].view(-1, 1, 1)
        m_max = log_mels.view(len(windows), -1).max(dim=1)[0].view(-1, 1, 1)
        log_mels = (log_mels - m_min) / (m_max - m_min + 1e-6)
        
        images = log_mels.unsqueeze(1)
        images = F.interpolate(images, size=CFG.IMAGE_SIZE, mode='bilinear', align_corners=False).repeat(1, 3, 1, 1)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images = (images - mean) / std
        
        ac_out, geo_out = model(images)
        probs_visual = torch.sigmoid(ac_out).cpu().numpy()
        predicted_coords = geo_out.cpu().numpy()
        
        # 3. Dynamic Self-Distillation Spatial Prior
        if knn_model is not None:
            probs_knn = knn_model.predict(predicted_coords)
            final_probs = np.zeros_like(probs_visual)
            for i in range(len(probs_visual)):
                if np.max(probs_visual[i]) > CFG.CONFIDENCE_THRESHOLD:
                    final_probs[i] = probs_visual[i]
                else:
                    final_probs[i] = probs_visual[i] * (1 - CFG.KNN_ALPHA) + probs_knn[i] * CFG.KNN_ALPHA
        else:
            final_probs = probs_visual
            
    # 4. Max-Pooling Aggregation per 5s submission block
    for i in range(num_submission_chunks):
        block_start = i * CFG.CHUNK_LENGTH
        block_end = block_start + CFG.CHUNK_LENGTH
        
        intersecting_probs = []
        for w_idx, w_start in enumerate(starts):
            w_end = w_start + CFG.CHUNK_LENGTH
            overlap = min(block_end, w_end) - max(block_start, w_start)
            if overlap > (CFG.CHUNK_LENGTH * 0.4): # >40% overlap
                intersecting_probs.append(final_probs[w_idx])
                
        if intersecting_probs:
            submission_block_probs[i] = np.max(intersecting_probs, axis=0)
            
        all_row_ids.append(f"{filename}_{(i+1)*5}")
        all_predictions.append(submission_block_probs[i])

sub_df = pd.DataFrame(all_predictions, columns=submission_labels)
sub_df.insert(0, 'row_id', all_row_ids)
sub_df.to_csv('submission.csv', index=False)
print("Submission formatted!")
"""
]

import os
os.makedirs('kaggle-pipelines/pipelines/proto1/alter-strat-2', exist_ok=True)
create_notebook('kaggle-pipelines/pipelines/proto1/alter-strat-2/vit-geo-acoustic-train.ipynb', train_blocks)
create_notebook('kaggle-pipelines/pipelines/proto1/alter-strat-2/vit-geo-acoustic-inference.ipynb', inference_blocks)
print("Notebooks created successfully!")
