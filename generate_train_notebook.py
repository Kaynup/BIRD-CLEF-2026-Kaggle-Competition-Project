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
            "source": [line + "\n" for line in code.split("\n")]
        })
    notebook = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python", "version": "3.10.0"}},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=1)

train_blocks = [
"""import os
import ast
import math
import glob
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import soundfile as sf
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import timm

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')

class CFG:
    # Path configuration for Kaggle environments
    ROOT_DIR = '/kaggle/input/competitions/birdclef-2026'
    TRAIN_CSV = os.path.join(ROOT_DIR, 'train.csv')
    SOUNDSCAPES_CSV = os.path.join(ROOT_DIR, 'train_soundscapes_labels.csv')
    SAMPLE_SUB_CSV = os.path.join(ROOT_DIR, 'sample_submission.csv')
    TRAIN_AUDIO_DIR = os.path.join(ROOT_DIR, 'train_audio')
    SOUNDSCAPE_DIR = os.path.join(ROOT_DIR, 'train_soundscapes')
    
    # Training Setup
    NUM_FOLDS = 5
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3
    NUM_WORKERS = 4
    
    # Audio Parameters
    SR = 32000
    DURATION_SECONDS = 5.0
    CHUNK_LENGTH = int(SR * DURATION_SECONDS)
    
    # Spectrogram setup
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512
    FMIN = 50
    FMAX = 16000
    
    # Backbone Parameters
    BACKBONE_NAME = 'vit_base_patch16_224'
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 234
    
    # Spatial constants
    PANTANAL_LAT_CENTER = -19.05
    PANTANAL_LON_CENTER = -56.75
    
    # Dual Head Configuration
    GEO_LOSS_WEIGHT = 5.0
    CONFIDENCE_THRESHOLD = 0.85
    KNN_ALPHA = 0.3
""",
"""def oversample_rare_species(df, min_occurrences=5):
    counts = df['primary_label'].value_counts()
    rare_species = counts[counts < min_occurrences].index
    print(f"Found {len(rare_species)} species with less than {min_occurrences} occurrences.")
    
    oversampled_rows = []
    for species in rare_species:
        species_rows = df[df['primary_label'] == species]
        n_existing = len(species_rows)
        if n_existing == 0: continue
        n_needed = min_occurrences - n_existing
        repeated = species_rows.sample(n=n_needed, replace=True)
        oversampled_rows.append(repeated)
        
    if oversampled_rows:
        df = pd.concat([df] + oversampled_rows, ignore_index=True)
    print(f"Oversampling completed. Total rows: {len(df)}")
    return df

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

    def _load_audio(self, path, frame_offset=0, num_frames=-1):
        try:
            data, sample_rate = sf.read(path, start=frame_offset, frames=num_frames, dtype='float32')
            if data.ndim > 1: data = data.mean(axis=1) # Mono
            if sample_rate != CFG.SR:
                resampler = T.Resample(sample_rate, CFG.SR)
                data = resampler(torch.tensor(data).unsqueeze(0)).squeeze(0).numpy()
        except:
            data = np.zeros(CFG.CHUNK_LENGTH)
            
        waveform = torch.tensor(data)
        total_len = waveform.size(0)
        
        # Random cropping for training short audio
        if num_frames == -1 and total_len > CFG.CHUNK_LENGTH:
            if self.is_train:
                offset = random.randint(0, total_len - CFG.CHUNK_LENGTH)
                waveform = waveform[offset:offset + CFG.CHUNK_LENGTH]
            else:
                offset = (total_len - CFG.CHUNK_LENGTH) // 2
                waveform = waveform[offset:offset + CFG.CHUNK_LENGTH]
        elif waveform.size(0) < CFG.CHUNK_LENGTH:
            pad_len = CFG.CHUNK_LENGTH - waveform.size(0)
            waveform = F.pad(waveform, (0, pad_len))
        elif waveform.size(0) > CFG.CHUNK_LENGTH:
            waveform = waveform[:CFG.CHUNK_LENGTH]
            
        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = torch.zeros(CFG.NUM_CLASSES, dtype=torch.float32)
        
        if 'start' in row and not pd.isna(row['start']):
            # Soundscape expert labeled data
            audio_path = os.path.join(CFG.SOUNDSCAPE_DIR, row['filename'])
            start_sec = float(row['start']) if isinstance(row['start'], (int, float)) else 0.0
            frame_offset = int(start_sec * CFG.SR)
            waveform = self._load_audio(audio_path, frame_offset=frame_offset, num_frames=CFG.CHUNK_LENGTH)
            
            labels = str(row['primary_label']).split(';')
            for label in labels:
                label = label.strip()
                if label in self.species_to_idx:
                    target[self.species_to_idx[label]] = 1.0
        else:
            # Short training audio
            audio_path = os.path.join(CFG.TRAIN_AUDIO_DIR, row['filename'])
            waveform = self._load_audio(audio_path)
            
            primary = str(row['primary_label']).strip()
            if primary in self.species_to_idx:
                target[self.species_to_idx[primary]] = 1.0
                
            try:
                secondaries = ast.literal_eval(row['secondary_labels'])
                for label in secondaries:
                    label = label.strip()
                    if label in self.species_to_idx:
                        target[self.species_to_idx[label]] = 0.5
            except:
                pass
                
        # Spectrogram processing
        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-6)
        
        if self.is_train: # Heavy background augmentation from 01-multilabel can be added here
            log_mel = self.time_masking(log_mel)
            log_mel = self.freq_masking(log_mel)
        
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
        image = log_mel.unsqueeze(0).unsqueeze(0)
        image = F.interpolate(image, size=CFG.IMAGE_SIZE, mode='bilinear', align_corners=False).squeeze(0).repeat(3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        # Geographic Coordinates
        lat = float(row.get('latitude', CFG.PANTANAL_LAT_CENTER))
        lon = float(row.get('longitude', CFG.PANTANAL_LON_CENTER))
        if np.isnan(lat): lat = CFG.PANTANAL_LAT_CENTER
        if np.isnan(lon): lon = CFG.PANTANAL_LON_CENTER
        coords = torch.tensor(np.radians([lat, lon]), dtype=torch.float32)
        
        return image, coords, target
""",
"""def optimize_and_train_spatial_knn(train_df, submission_labels, species_to_idx):
    print("\\nInitializing Spatial KNN optimization...")
    spatial_df = train_df.dropna(subset=['latitude', 'longitude']).copy()
    X_rad = np.radians(spatial_df[['latitude', 'longitude']].values)
    
    y_targets = np.zeros((len(spatial_df), len(submission_labels)), dtype=np.float32)
    for idx, row in enumerate(spatial_df.itertuples()):
        primary = str(row.primary_label).strip()
        if primary in species_to_idx: y_targets[idx, species_to_idx[primary]] = 1.0
        try:
            for label in ast.literal_eval(row.secondary_labels):
                label = label.strip()
                if label in species_to_idx: y_targets[idx, species_to_idx[label]] = 0.5
        except: pass
            
    best_neighbors = 15
    best_weights = 'distance'
    best_score = -1.0
    
    split_idx = int(len(X_rad) * 0.8)
    X_tr, X_val = X_rad[:split_idx], X_rad[split_idx:]
    y_tr, y_val = y_targets[:split_idx], y_targets[split_idx:]
    
    for k in [5, 15, 30, 50]:
        for w in ['uniform', 'distance']:
            knn = KNeighborsRegressor(n_neighbors=k, weights=w, metric='haversine')
            knn.fit(X_tr, y_tr)
            preds = knn.predict(X_val)
            valid_scores = []
            for col in range(y_val.shape[1]):
                y_true_col = (y_val[:, col] > 0).astype(int)
                if len(np.unique(y_true_col)) > 1:
                    valid_scores.append(roc_auc_score(y_true_col, preds[:, col]))
            score = np.mean(valid_scores) if valid_scores else 0.5
            if score > best_score:
                best_score = score; best_neighbors = k; best_weights = w
                
    print(f"Optimal KNN parameters found: n_neighbors={best_neighbors}, weights='{best_weights}' (Val Score: {best_score:.4f})")
    final_knn = KNeighborsRegressor(n_neighbors=best_neighbors, weights=best_weights, metric='haversine')
    final_knn.fit(X_rad, y_targets)
    return final_knn

class DualHeadViT(nn.Module):
    def __init__(self, backbone_name=CFG.BACKBONE_NAME, num_classes=CFG.NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, in_chans=3)
        in_features = self.backbone.head.in_features if hasattr(self.backbone, 'head') else 768
        self.backbone.reset_classifier(0)
        
        self.acoustic_head = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_classes)
        )
        self.geo_head = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 2)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.acoustic_head(features), self.geo_head(features)
""",
"""def train_one_epoch(model, dataloader, optimizer, crit_ac, crit_geo, device):
    model.train()
    total_loss = 0.0
    all_tr_targets, all_tr_preds = [], []
    
    for images, geo_targets, targets in tqdm(dataloader, desc="Training Epoch"):
        images, targets, geo_targets = images.to(device), targets.to(device), geo_targets.to(device)
        
        optimizer.zero_grad()
        ac_out, geo_out = model(images)
        loss = crit_ac(ac_out, targets) + (CFG.GEO_LOSS_WEIGHT * crit_geo(geo_out, geo_targets))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        all_tr_targets.append(targets.cpu().numpy())
        all_tr_preds.append(torch.sigmoid(ac_out).detach().cpu().numpy())
        
    all_tr_targets = np.concatenate(all_tr_targets, axis=0)
    all_tr_preds = np.concatenate(all_tr_preds, axis=0)
    
    train_auc_scores = []
    for col in range(CFG.NUM_CLASSES):
        true_s = (all_tr_targets[:, col] > 0).astype(int)
        pred_s = all_tr_preds[:, col]
        if len(np.unique(true_s)) > 1: train_auc_scores.append(roc_auc_score(true_s, pred_s))
    mean_train_auc = np.mean(train_auc_scores) if train_auc_scores else 0.5
    
    return total_loss / len(dataloader), mean_train_auc

@torch.no_grad()
def validate(model, dataloader, knn_model, crit_ac, crit_geo, submission_labels, device, epoch):
    model.eval()
    total_loss = 0.0
    all_targets, all_preds = [], []
    
    for images, geo_targets, targets in tqdm(dataloader, desc="Validation Epoch"):
        images, targets, geo_targets = images.to(device), targets.to(device), geo_targets.to(device)
        
        ac_out, geo_out = model(images)
        loss = crit_ac(ac_out, targets) + (CFG.GEO_LOSS_WEIGHT * crit_geo(geo_out, geo_targets))
        total_loss += loss.item()
        
        probs_visual = torch.sigmoid(ac_out).cpu().numpy()
        predicted_coords = geo_out.cpu().numpy()
        probs_knn = knn_model.predict(predicted_coords)
        
        final_probs = np.zeros_like(probs_visual)
        for i in range(len(probs_visual)):
            if np.max(probs_visual[i]) > CFG.CONFIDENCE_THRESHOLD:
                final_probs[i] = probs_visual[i]
            else:
                final_probs[i] = probs_visual[i] * (1 - CFG.KNN_ALPHA) + probs_knn[i] * CFG.KNN_ALPHA
                
        all_targets.append(targets.cpu().numpy())
        all_preds.append(final_probs)
        
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    binary_targets = (all_targets > 0).astype(int)
    binary_preds = (all_preds >= 0.5).astype(int)
    
    val_f1 = f1_score(binary_targets, binary_preds, average='macro', zero_division=0)
    val_precision = precision_score(binary_targets, binary_preds, average='macro', zero_division=0)
    val_recall = recall_score(binary_targets, binary_preds, average='macro', zero_division=0)
    
    valid_auc_scores = []
    for idx in range(CFG.NUM_CLASSES):
        true_s = (all_targets[:, idx] > 0).astype(int)
        pred_s = all_preds[:, idx]
        if len(np.unique(true_s)) > 1: valid_auc_scores.append(roc_auc_score(true_s, pred_s))
    mean_macro_auc = np.mean(valid_auc_scores) if valid_auc_scores else 0.5
    
    return total_loss / len(dataloader), mean_macro_auc, val_f1, val_precision, val_recall
""",
"""def print_metrics_table(epoch, train_loss, train_auc, val_loss, val_auc, val_f1, val_precision, val_recall, generalization_score):
    loss_diff = val_loss - train_loss
    auc_diff = val_auc - train_auc
    print("\\n" + "="*65)
    print(f"|                  EPOCH {epoch:02d} PERFORMANCE REPORT                   |")
    print("="*65)
    print(f"| {'Metric':<20} | {'Train':<10} | {'Validation':<12} | {'Contrast':<10} |")
    print("-" * 65)
    print(f"| {'Loss':<20} | {train_loss:<10.4f} | {val_loss:<12.4f} | {loss_diff:<+10.4f} |")
    print(f"| {'Macro-AUC':<20} | {train_auc:<10.4f} | {val_auc:<12.4f} | {auc_diff:<+10.4f} |")
    print(f"| {'Generalization (AUC)':<20} | {'N/A':<10} | {generalization_score:<12.4f} | {'-':<10} |")
    print("-" * 65)
    print(f"| {'F1-Macro':<20} | {'N/A':<10} | {val_f1:<12.4f} | {'-':<10} |")
    print(f"| {'Precision-Macro':<20} | {'N/A':<10} | {val_precision:<12.4f} | {'-':<10} |")
    print(f"| {'Recall-Macro':<20} | {'N/A':<10} | {val_recall:<12.4f} | {'-':<10} |")
    print("="*65 + "\\n")

def generate_and_save_auditing_plots(history_df, fold):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    epochs = history_df['epoch'].values
    
    axes[0].plot(epochs, history_df['train_loss'].values, label='Train Loss', marker='o')
    axes[0].plot(epochs, history_df['val_loss'].values, label='Val Loss', marker='s')
    axes[0].set_title('Loss Convergence'); axes[0].legend(); axes[0].grid(True)
    
    axes[1].plot(epochs, history_df['train_auc'].values, label='Train AUC', marker='^')
    axes[1].plot(epochs, history_df['val_auc'].values, label='Val AUC', marker='v')
    axes[1].plot(epochs, history_df['generalization_score'].values, label='Gen Score', marker='d', linestyle='--')
    axes[1].set_title('AUC & Generalization'); axes[1].legend(); axes[1].grid(True)
    
    axes[2].plot(epochs, history_df['val_f1'].values, label='Val F1', marker='x')
    axes[2].plot(epochs, history_df['val_precision'].values, label='Val Precision', marker='+')
    axes[2].plot(epochs, history_df['val_recall'].values, label='Val Recall', marker='*')
    axes[2].set_title('Decision Boundary Metrics'); axes[2].legend(); axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"fold_{fold}_performance_audit.png", dpi=300)
    plt.close()
""",
"""def main():
    print("Initializing dataset and labels...")
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    soundscapes_df = pd.read_csv(CFG.SOUNDSCAPES_CSV)
    sample_sub = pd.read_csv(CFG.SAMPLE_SUB_CSV)
    
    submission_labels = [c for c in sample_sub.columns if c != 'row_id']
    species_to_idx = {species: idx for idx, species in enumerate(submission_labels)}
    CFG.NUM_CLASSES = len(submission_labels)
    
    # 1. OVER-SAMPLING TO ENABLE PERFECT STRATIFICATION
    train_df = oversample_rare_species(train_df, min_occurrences=5)
    
    # 2. ZERO-LEAKAGE STRATIFICATION (Group by author)
    train_df['author'] = train_df['author'].fillna('unknown')
    train_df['stratify_label'] = train_df['primary_label'].apply(lambda x: str(x).split(';')[0])
    sgkf = StratifiedGroupKFold(n_splits=CFG.NUM_FOLDS)
    folds = list(sgkf.split(train_df, train_df['stratify_label'], groups=train_df['author']))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold in range(1):
        print(f"\\n{'='*20} Starting Fold {fold} Training {'='*20}")
        train_idx, val_idx = folds[fold]
        fold_train_df = train_df.iloc[train_idx].copy()
        fold_val_df = train_df.iloc[val_idx].copy()
        
        # 3. KNN OPTIMIZATION (Dynamic Search)
        knn_spatial_model = optimize_and_train_spatial_knn(fold_train_df, submission_labels, species_to_idx)
        joblib.dump(knn_spatial_model, f"knn_spatial_fold_{fold}.pkl")
        
        train_dataset = BirdCLEFDataset(fold_train_df, species_to_idx, is_train=True)
        val_dataset = BirdCLEFDataset(fold_val_df, species_to_idx, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)
        
        model = DualHeadViT(num_classes=CFG.NUM_CLASSES).to(device)
        criterion_ac = nn.BCEWithLogitsLoss()
        criterion_geo = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
        
        epoch_history = []
        best_gen_score = -1.0
        
        for epoch in range(1, CFG.EPOCHS + 1):
            train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion_ac, criterion_geo, device)
            val_loss, val_auc, val_f1, val_precision, val_recall = validate(model, val_loader, knn_spatial_model, criterion_ac, criterion_geo, submission_labels, device, epoch)
            
            # 4. ROBUST GENERALIZATION CRITERION
            generalization_score = val_auc - max(0.0, train_auc - val_auc)
            
            # 5. EPOCH WEIGHT SAVING
            checkpoint_name = f"model_vit_fold_{fold}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_name)
            
            epoch_history.append({
                'epoch': epoch, 'train_loss': train_loss, 'train_auc': train_auc,
                'val_loss': val_loss, 'val_auc': val_auc, 'val_f1': val_f1,
                'val_precision': val_precision, 'val_recall': val_recall,
                'generalization_score': generalization_score, 'saved_weights': checkpoint_name
            })
            
            pd.DataFrame(epoch_history).to_csv(f"fold_{fold}_training_history.csv", index=False)
            print_metrics_table(epoch, train_loss, train_auc, val_loss, val_auc, val_f1, val_precision, val_recall, generalization_score)
            
            if generalization_score > best_gen_score:
                best_gen_score = generalization_score
                torch.save(model.state_dict(), f"best_model_vit_fold_{fold}.pth")
                print(f"   => New best generalization model checkpoint saved! (Gen Score: {generalization_score:.4f})")
                
        generate_and_save_auditing_plots(pd.DataFrame(epoch_history), fold)

if __name__ == '__main__':
    main()
"""
]

create_notebook('kaggle-pipelines/pipelines/proto1/alter-strat-2/vit-geo-acoustic-train.ipynb', train_blocks)
