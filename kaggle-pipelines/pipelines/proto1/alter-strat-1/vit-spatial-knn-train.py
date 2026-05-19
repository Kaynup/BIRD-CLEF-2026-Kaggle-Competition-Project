import os
import ast
import math
import time
import glob
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import timm

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score

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
    BACKBONE_PATH = '/kaggle/input/models/fp3924/vit_base_patch16_224/pytorch/default/1/vit_base_patch16_224.pth'
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 234
    
    # Spatial constants
    PANTANAL_LAT_CENTER = -19.05
    PANTANAL_LON_CENTER = -56.75
    
    # Dynamic Blend Alpha (weight of KNN spatial predictions vs ViT predictions)
    SPATIAL_BLEND_ALPHA = 0.2

print("Global pipeline configurations loaded successfully.")

def oversample_rare_species(df, min_occurrences=5):
    """
    Oversamples rare species so that every species has at least min_occurrences
    allowing perfect 5-fold stratification across validation sets.
    """
    counts = df['primary_label'].value_counts()
    rare_species = counts[counts < min_occurrences].index
    print(f"Found {len(rare_species)} species with less than {min_occurrences} occurrences.")
    
    oversampled_rows = []
    for species in rare_species:
        species_rows = df[df['primary_label'] == species]
        n_existing = len(species_rows)
        if n_existing == 0:
            continue
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
        
        # Audio transform setup
        self.mel_transform = T.MelSpectrogram(
            sample_rate=CFG.SR,
            n_fft=CFG.N_FFT,
            hop_length=CFG.HOP_LENGTH,
            n_mels=CFG.N_MELS,
            f_min=CFG.FMIN,
            f_max=CFG.FMAX
        )

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path, frame_offset=0, num_frames=-1):
        try:
            data, sample_rate = sf.read(path, start=frame_offset, frames=num_frames, dtype='float32')
            if data.ndim == 1:
                waveform = torch.tensor(data).unsqueeze(0)
            else:
                waveform = torch.tensor(data).T
            if sample_rate != CFG.SR:
                resampler = T.Resample(sample_rate, CFG.SR)
                waveform = resampler(waveform)
        except:
            waveform = torch.zeros((1, CFG.CHUNK_LENGTH))
        
        # Downmix to Mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad/Crop
        if waveform.size(1) < CFG.CHUNK_LENGTH:
            pad_len = CFG.CHUNK_LENGTH - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_len))
        elif waveform.size(1) > CFG.CHUNK_LENGTH:
            waveform = waveform[:, :CFG.CHUNK_LENGTH]
            
        return waveform.squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if 'start' in row and not pd.isna(row['start']):
            # Soundscape expert labeled data
            audio_path = os.path.join(CFG.SOUNDSCAPE_DIR, row['filename'])
            # start is typically given in seconds or HH:MM:SS
            start_sec = float(row['start']) if isinstance(row['start'], (int, float)) else 0.0
            frame_offset = int(start_sec * CFG.SR)
            waveform = self._load_audio(audio_path, frame_offset=frame_offset, num_frames=CFG.CHUNK_LENGTH)
            
            target = torch.zeros(CFG.NUM_CLASSES, dtype=torch.float32)
            labels = str(row['primary_label']).split(';')
            for label in labels:
                label = label.strip()
                if label in self.species_to_idx:
                    target[self.species_to_idx[label]] = 1.0
        else:
            # Short training audio
            audio_path = os.path.join(CFG.TRAIN_AUDIO_DIR, row['filename'])
            waveform = self._load_audio(audio_path, num_frames=CFG.CHUNK_LENGTH)
            
            target = torch.zeros(CFG.NUM_CLASSES, dtype=torch.float32)
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
                
        # Extract Mel-spectrogram & Resize dynamically to fit ViT 224x224 input
        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-6)
        
        # Scale to 0-1
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
        
        # Dynamic resize to fit Vision Transformer's expected 224x224 input
        image = log_mel.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, H, W]
        image = F.interpolate(image, size=CFG.IMAGE_SIZE, mode='bilinear', align_corners=False)
        image = image.squeeze(0).repeat(3, 1, 1)  # Expand to 3 channels: [3, 224, 224]
        
        # Standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        # Return geographic coordinates alongside waveform representation
        lat = float(row.get('latitude', CFG.PANTANAL_LAT_CENTER))
        lon = float(row.get('longitude', CFG.PANTANAL_LON_CENTER))
        if np.isnan(lat): lat = CFG.PANTANAL_LAT_CENTER
        if np.isnan(lon): lon = CFG.PANTANAL_LON_CENTER
        
        coords = torch.tensor([lat, lon], dtype=torch.float32)
        
        return image, coords, target

def optimize_and_train_spatial_knn(train_df, submission_labels, species_to_idx):
    """
    Trains a Geographic KNN regressor in radians to map coordinates
    directly to species density priors, with automated hyperparameter auto-tuning.
    """
    print("\nInitializing Spatial KNN optimization...")
    spatial_df = train_df.dropna(subset=['latitude', 'longitude']).copy()
    
    # Convert degrees to radians for robust Haversine metric
    X_rad = np.radians(spatial_df[['latitude', 'longitude']].values)
    
    # Build multi-label Target targets
    y_targets = np.zeros((len(spatial_df), len(submission_labels)), dtype=np.float32)
    for idx, row in enumerate(spatial_df.itertuples()):
        # Foreground
        primary = str(row.primary_label).strip()
        if primary in species_to_idx:
            y_targets[idx, species_to_idx[primary]] = 1.0
        # Background
        try:
            secondaries = ast.literal_eval(row.secondary_labels)
            for label in secondaries:
                label = label.strip()
                if label in species_to_idx:
                    y_targets[idx, species_to_idx[label]] = 0.5
        except:
            pass
            
    # Hyperparameter Grid Search (Auto-improving settings)
    best_neighbors = 15
    best_weights = 'distance'
    best_score = -1.0
    
    # Search candidates
    neighbors_candidates = [5, 15, 30, 50]
    weights_candidates = ['uniform', 'distance']
    
    # Fast validation search
    split_idx = int(len(X_rad) * 0.8)
    X_tr, X_val = X_rad[:split_idx], X_rad[split_idx:]
    y_tr, y_val = y_targets[:split_idx], y_targets[split_idx:]
    
    for k in neighbors_candidates:
        for w in weights_candidates:
            knn = KNeighborsRegressor(n_neighbors=k, weights=w, metric='haversine')
            knn.fit(X_tr, y_tr)
            preds = knn.predict(X_val)
            # Calculate simple multi-label correlation validation metric with binarized targets
            valid_scores = []
            for col in range(y_val.shape[1]):
                y_true_col = (y_val[:, col] > 0).astype(int)
                if len(np.unique(y_true_col)) > 1:
                    valid_scores.append(roc_auc_score(y_true_col, preds[:, col]))
            score = np.mean(valid_scores) if valid_scores else 0.5
            if score > best_score:
                best_score = score
                best_neighbors = k
                best_weights = w
                
    print(f"Optimal KNN parameters found: n_neighbors={best_neighbors}, weights='{best_weights}' (Val Score: {best_score:.4f})")
    
    # Train final KNN on complete dataset
    final_knn = KNeighborsRegressor(n_neighbors=best_neighbors, weights=best_weights, metric='haversine')
    final_knn.fit(X_rad, y_targets)
    return final_knn

class BirdViTModel(nn.Module):
    def __init__(self, backbone_name=CFG.BACKBONE_NAME, backbone_path=CFG.BACKBONE_PATH, num_classes=CFG.NUM_CLASSES):
        super().__init__()
        print(f"Initializing pre-trained Vision Transformer backbone: {backbone_name}")
        self.backbone = timm.create_model(backbone_name, pretrained=False, checkpoint_path=backbone_path, in_chans=3)
        
        # Extract and reset final classifier layer
        in_features = self.backbone.head.in_features if hasattr(self.backbone, 'head') else 768
        self.backbone.reset_classifier(0)
        
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Forward through Vision Transformer backbone to extract visual features
        features = self.backbone(x)
        out = self.head(features)
        return out

from sklearn.metrics import f1_score, precision_score, recall_score

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    all_tr_targets = []
    all_tr_preds = []
    
    for images, _, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Record outputs and targets for epoch-level Train AUC
        all_tr_targets.append(targets.cpu().numpy())
        all_tr_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        
    avg_loss = total_loss / len(dataloader)
    all_tr_targets = np.concatenate(all_tr_targets, axis=0)
    all_tr_preds = np.concatenate(all_tr_preds, axis=0)
    
    # Calculate epoch-level Train AUC over active classes
    train_auc_scores = []
    for col in range(CFG.NUM_CLASSES):
        true_s = (all_tr_targets[:, col] > 0).astype(int)
        pred_s = all_tr_preds[:, col]
        if len(np.unique(true_s)) > 1:
            train_auc_scores.append(roc_auc_score(true_s, pred_s))
    mean_train_auc = np.mean(train_auc_scores) if train_auc_scores else 0.5
    
    return avg_loss, mean_train_auc

@torch.no_grad()
def validate(model, dataloader, knn_model, criterion, submission_labels, species_to_idx, device, epoch):
    model.eval()
    total_loss = 0.0
    
    all_targets = []
    all_preds = []
    
    for images, coords, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        
        # Get visual prediction probabilities
        probs_visual = torch.sigmoid(outputs).cpu().numpy()
        
        # Retrieve spatial priors from hyperparameter optimized KNN model in radians
        coords_rad = np.radians(coords.numpy())
        probs_knn = knn_model.predict(coords_rad)
        
        # Perform weighted prior blending
        final_probs = probs_visual * (1.0 - CFG.SPATIAL_BLEND_ALPHA) + probs_knn * CFG.SPATIAL_BLEND_ALPHA
        
        all_targets.append(targets.cpu().numpy())
        all_preds.append(final_probs)
        
    avg_loss = total_loss / len(dataloader)
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    # Binarize validation targets and predictions for multiclass scores
    binary_targets = (all_targets > 0).astype(int)
    binary_preds = (all_preds >= 0.5).astype(int)
    
    # Calculate macro-averaged precision, recall, and F1 scores
    val_f1 = f1_score(binary_targets, binary_preds, average='macro', zero_division=0)
    val_precision = precision_score(binary_targets, binary_preds, average='macro', zero_division=0)
    val_recall = recall_score(binary_targets, binary_preds, average='macro', zero_division=0)
    
    # Calculate class-wise AUC for every single species
    species_auc = {}
    valid_auc_scores = []
    for idx, species in enumerate(submission_labels):
        true_s = (all_targets[:, idx] > 0).astype(int)
        pred_s = all_preds[:, idx]
        
        if len(np.unique(true_s)) > 1:
            auc = roc_auc_score(true_s, pred_s)
            species_auc[species] = auc
            valid_auc_scores.append(auc)
        else:
            species_auc[species] = np.nan
            
    # Write per-species metrics to CSV for analysis
    species_df = pd.DataFrame(list(species_auc.items()), columns=['species', 'auc'])
    species_df.to_csv(f"species_metrics_epoch_{epoch}.csv", index=False)
    
    mean_macro_auc = np.mean(valid_auc_scores) if valid_auc_scores else 0.5
    return avg_loss, mean_macro_auc, val_f1, val_precision, val_recall

import matplotlib.pyplot as plt
import joblib

def print_metrics_table(epoch, train_loss, train_auc, val_loss, val_auc, val_f1, val_precision, val_recall, generalization_score):
    loss_diff = val_loss - train_loss
    auc_diff = val_auc - train_auc
    
    print("\n" + "="*65)
    print(f"|                  EPOCH {epoch:02d} PERFORMANCE REPORT                   |")
    print("="*65)
    print(f"| {'Metric':<20} | {'Train':<10} | {'Validation':<12} | {'Contrast':<10} |")
    print("-"*65)
    print(f"| {'Loss':<20} | {train_loss:<10.4f} | {val_loss:<12.4f} | {loss_diff:<+10.4f} |")
    print(f"| {'Macro-AUC':<20} | {train_auc:<10.4f} | {val_auc:<12.4f} | {auc_diff:<+10.4f} |")
    print(f"| {'Generalization':<20} | {'N/A':<10} | {generalization_score:<12.4f} | {'-':<10} |")
    print("-"*65)
    print(f"| {'F1-Macro':<20} | {'N/A':<10} | {val_f1:<12.4f} | {'-':<10} |")
    print(f"| {'Precision-Macro':<20} | {'N/A':<10} | {val_precision:<12.4f} | {'-':<10} |")
    print(f"| {'Recall-Macro':<20} | {'N/A':<10} | {val_recall:<12.4f} | {'-':<10} |")
    print("="*65 + "\n")

def generate_and_save_auditing_plots(history_df, fold):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    epochs = history_df['epoch'].values
    
    # 1. Loss Panel
    axes[0].plot(epochs, history_df['train_loss'].values, label='Train Loss', color='#1f77b4', marker='o')
    axes[0].plot(epochs, history_df['val_loss'].values, label='Val Loss', color='#ff7f0e', marker='s')
    axes[0].set_title('Loss Convergence', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. AUC & Generalization Panel
    axes[1].plot(epochs, history_df['train_auc'].values, label='Train AUC', color='#2ca02c', marker='^')
    axes[1].plot(epochs, history_df['val_auc'].values, label='Val AUC', color='#d62728', marker='v')
    axes[1].plot(epochs, history_df['generalization_score'].values, label='Gen Score', color='#9467bd', marker='d', linestyle='--')
    axes[1].set_title('AUC & Generalization Audit', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. Decision Boundary Metrics Panel
    axes[2].plot(epochs, history_df['val_f1'].values, label='Val F1', color='#bcbd22', marker='x')
    axes[2].plot(epochs, history_df['val_precision'].values, label='Val Precision', color='#17becf', marker='+')
    axes[2].plot(epochs, history_df['val_recall'].values, label='Val Recall', color='#e377c2', marker='*')
    axes[2].set_title('Decision Boundary Metrics', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Score')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"fold_{fold}_performance_audit.png", dpi=300)
    plt.close()
    print(f"Auditing plots generated and saved successfully to: fold_{fold}_performance_audit.png")

def main():
    print("Initializing dataset and labels...")
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    soundscapes_df = pd.read_csv(CFG.SOUNDSCAPES_CSV)
    sample_sub = pd.read_csv(CFG.SAMPLE_SUB_CSV)
    
    submission_labels = [c for c in sample_sub.columns if c != 'row_id']
    species_to_idx = {species: idx for idx, species in enumerate(submission_labels)}
    
    # Oversample rare labels with less than 5 occurrences to enable proper stratification
    train_df = oversample_rare_species(train_df, min_occurrences=5)
    
    # Perform 5-fold stratification
    sgkf = StratifiedGroupKFold(n_splits=CFG.NUM_FOLDS)
    train_df['stratify_label'] = train_df['primary_label'].apply(lambda x: str(x).split(';')[0])
    folds = list(sgkf.split(train_df, train_df['stratify_label'], groups=train_df['filename']))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Execute Fold 0 Prototype Training
    for fold in range(1):
        print(f"\n{'='*20} Starting Fold {fold} Training {'='*20}")
        train_idx, val_idx = folds[fold]
        
        fold_train_df = train_df.iloc[train_idx].copy()
        fold_val_df = train_df.iloc[val_idx].copy()
        
        # 1. Optimize and train the Spatial KNN Regressor in Radians
        knn_spatial_model = optimize_and_train_spatial_knn(fold_train_df, submission_labels, species_to_idx)
        
        # Save optimized KNN spatial prior
        knn_name = f"knn_spatial_fold_{fold}.pkl"
        joblib.dump(knn_spatial_model, knn_name)
        print(f"   Saved pre-trained KNN spatial model: {knn_name}")
        
        # 2. Instantiate Dataset and DataLoaders
        train_dataset = BirdCLEFDataset(fold_train_df, species_to_idx, is_train=True)
        val_dataset = BirdCLEFDataset(fold_val_df, species_to_idx, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)
        
        # 3. Instantiate ViT Model & Optimizer
        model = BirdViTModel(backbone_name=CFG.BACKBONE_NAME, num_classes=CFG.NUM_CLASSES).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
        
        epoch_history = []
        best_gen_score = -1.0
        
        for epoch in range(1, CFG.EPOCHS + 1):
            print(f"Epoch {epoch}/{CFG.EPOCHS}")
            train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_auc, val_f1, val_precision, val_recall = validate(model, val_loader, knn_spatial_model, criterion, submission_labels, species_to_idx, device, epoch)
            
            # Calculate Generalization Score (penalized by overfitting gap)
            generalization_score = val_auc - max(0.0, train_auc - val_auc)
            
            # Save weights for EVERY epoch to allow custom epoch auditing/selection
            checkpoint_name = f"model_vit_fold_{fold}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_name)
            
            # Record epoch metadata
            epoch_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_auc': train_auc,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'generalization_score': generalization_score,
                'saved_weights': checkpoint_name
            })
            
            # Save running metrics CSV history
            history_df = pd.DataFrame(epoch_history)
            history_df.to_csv(f"fold_{fold}_training_history.csv", index=False)
            
            # Print premium, easily readable tabular metrics
            print_metrics_table(epoch, train_loss, train_auc, val_loss, val_auc, val_f1, val_precision, val_recall, generalization_score)
            
            if generalization_score > best_gen_score:
                best_gen_score = generalization_score
                torch.save(model.state_dict(), f"best_model_vit_fold_{fold}.pth")
                print(f"   New best generalization model checkpoint saved successfully! (Gen Score: {generalization_score:.4f})")
        
        # Generate and save diagnostic auditing plots at the end of training
        generate_and_save_auditing_plots(history_df, fold)

if __name__ == '__main__':
    main()

