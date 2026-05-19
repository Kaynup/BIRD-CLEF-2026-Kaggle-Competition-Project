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
import torchaudio.transforms as T
import timm

from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')

class CFG:
    ROOT_DIR = '/kaggle/input/competitions/birdclef-2026'
    TRAIN_CSV = os.path.join(ROOT_DIR, 'train.csv')
    SAMPLE_SUB_CSV = os.path.join(ROOT_DIR, 'sample_submission.csv')
    SOUNDSCAPE_DIR = os.path.join(ROOT_DIR, 'test_soundscapes')
    
    # Point this to your trained ViT model weights and KNN pickle
    MODEL_PATH = '/kaggle/input/models/punyakdei/training-mol/pytorch/default/1/model_vit_fold_0_epoch_4.pth'
    KNN_PATH = '/kaggle/input/models/punyakdei/training-mol/pytorch/default/1/knn_spatial_fold_0.pkl'
    
    # Audio Setup
    SR = 32000
    WINDOW_SECONDS = 5.0
    CHUNK_LENGTH = int(SR * WINDOW_SECONDS)
    
    # Spectrogram setup
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512
    FMIN = 50
    FMAX = 16000
    
    # Model Parameters
    BACKBONE_NAME = 'vit_base_patch16_224'
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 234
    
    # Geographic Center
    PANTANAL_LAT_CENTER = -19.05
    PANTANAL_LON_CENTER = -56.75
    SPATIAL_BLEND_ALPHA = 0.2

CFG = Config = CFG()

print("Loading submission columns and mapping indexes...")
sample_sub = pd.read_csv(CFG.SAMPLE_SUB_CSV)
submission_labels = [c for c in sample_sub.columns if c != 'row_id']
CFG.NUM_CLASSES = len(submission_labels)
species_to_idx = {species: idx for idx, species in enumerate(submission_labels)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inference Hardware: {device}")

class BirdViTModel(nn.Module):
    def __init__(self, backbone_name=CFG.BACKBONE_NAME, num_classes=CFG.NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, in_chans=3)
        in_features = self.backbone.head.in_features if hasattr(self.backbone, 'head') else 768
        self.backbone.reset_classifier(0)
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out

def train_spatial_knn_inference(train_csv_path, submission_labels, species_to_idx):
    """
    Quick-trains the optimized KNN regressor on coordinates during inference startup.
    Uses a blazing-fast string splitter instead of slow ast.literal_eval.
    """
    print("Quick-training Spatial KNN prior learner...")
    spatial_df = pd.read_csv(train_csv_path).dropna(subset=['latitude', 'longitude']).copy()
    X_rad = np.radians(spatial_df[['latitude', 'longitude']].values)
    
    y_targets = np.zeros((len(spatial_df), len(submission_labels)), dtype=np.float32)
    for idx, row in enumerate(spatial_df.itertuples()):
        # Foreground
        primary = str(row.primary_label).strip()
        if primary in species_to_idx:
            y_targets[idx, species_to_idx[primary]] = 1.0
        
        # Blazing-fast string split parser for secondary labels
        label_str = str(row.secondary_labels).strip()
        if label_str and label_str != '[]':
            clean_str = label_str.strip('[]').replace("'", "").replace('"', "")
            secondaries = [lbl.strip() for lbl in clean_str.split(',') if lbl.strip()]
            for label in secondaries:
                if label in species_to_idx:
                    y_targets[idx, species_to_idx[label]] = 0.5
            
    # Match optimized grid search settings
    knn = KNeighborsRegressor(n_neighbors=50, weights='uniform', metric='haversine')
    knn.fit(X_rad, y_targets)
    return knn

import joblib

# 1. Load pre-trained ViT Model strictly
print("Instantiating ViT Model...")
model = BirdViTModel(backbone_name=CFG.BACKBONE_NAME, num_classes=CFG.NUM_CLASSES).to(device)
print(f"Loading strict weights from: {CFG.MODEL_PATH}")
model.load_state_dict(torch.load(CFG.MODEL_PATH, map_location=device))
model.eval()
print("ViT Model loaded successfully!")

# 2. Load pre-trained Spatial KNN model with bulletproof fallback
if os.path.exists(CFG.KNN_PATH):
    print(f"Loading pre-trained KNN model from: {CFG.KNN_PATH}...")
    knn_spatial_model = joblib.load(CFG.KNN_PATH)
    print("Pre-trained Spatial KNN model loaded successfully in 0.001 seconds!")
else:
    print("Pre-trained KNN model not found! Falling back to fast on-the-fly coordinate fitting...")
    knn_spatial_model = train_spatial_knn_inference(CFG.TRAIN_CSV, submission_labels, species_to_idx)
    print("Spatial KNN prior successfully trained on-the-fly!")

# 3. Resolve test files with fallback
test_files = sorted(glob.glob(f"{CFG.SOUNDSCAPE_DIR}/*.ogg"))
if len(test_files) == 0:
    print("Fallback Active: No test files found. Dry running with training soundscapes.")
    fallback_dir = os.path.join(CFG.ROOT_DIR, 'train_soundscapes')
    test_files = sorted(glob.glob(f"{fallback_dir}/*.ogg"))[:3]
else:
    print(f"Found {len(test_files)} test files to process.")

# 4. Sliding Window Inference
mel_transform = T.MelSpectrogram(
    sample_rate=CFG.SR, n_fft=CFG.N_FFT, hop_length=CFG.HOP_LENGTH,
    n_mels=CFG.N_MELS, f_min=CFG.FMIN, f_max=CFG.FMAX
).to(device)

# Pre-calculate KNN spatial prior for Pantanal recording center (Runs strictly ONCE)
pantanal_coord_rad = np.radians([[CFG.PANTANAL_LAT_CENTER, CFG.PANTANAL_LON_CENTER]])
probs_knn_prior = knn_spatial_model.predict(pantanal_coord_rad).squeeze(0)  # Shape [234]

all_predictions = []
all_row_ids = []

print(f"\n{'='*20} Executing ViT-Spatial-KNN Parallel Batch Inference {'='*20}")
for audio_path in tqdm(test_files, desc="Evaluating Audio Channels"):
    filename = os.path.basename(audio_path).replace('.ogg', '')
    
    try:
        data, sample_rate = sf.read(audio_path, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)  # Mono downmix
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        continue
        
    total_samples = len(data)
    chunk_len = CFG.CHUNK_LENGTH
    
    # Calculate exact segments and pad remaining boundary
    n_segments = math.ceil(total_samples / chunk_len)
    pad_len = (n_segments * chunk_len) - total_samples
    if pad_len > 0:
        data = np.pad(data, (0, pad_len))
        
    # Batch waveform loading directly to GPU
    waveforms = torch.tensor(data, dtype=torch.float32).reshape(n_segments, chunk_len).to(device)
    
    # Batched Spectrogram Generation
    with torch.no_grad():
        mel_specs = mel_transform(waveforms)  # Shape: [n_segments, N_MELS, Time]
        log_mels = torch.log(mel_specs + 1e-6)
        
        # Normalize each spectrogram individually in batch (using reshape for contiguous safety)
        m_min = log_mels.reshape(n_segments, -1).min(dim=1)[0].reshape(n_segments, 1, 1)
        m_max = log_mels.reshape(n_segments, -1).max(dim=1)[0].reshape(n_segments, 1, 1)
        log_mels = (log_mels - m_min) / (m_max - m_min + 1e-6)
        
        # Batched resizing to ViT 224x224 input
        images = log_mels.unsqueeze(1)  # Shape: [n_segments, 1, H, W]
        images = F.interpolate(images, size=CFG.IMAGE_SIZE, mode='bilinear', align_corners=False)
        images = images.repeat(1, 3, 1, 1)  # Expand to 3 channels: [n_segments, 3, 224, 224]
        
        # Batched ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)
        images = (images - mean) / std
        
        # Sub-batched ViT evaluation to prevent GPU VRAM overflow
        probs_visual_list = []
        sub_batch_size = 32
        for i in range(0, n_segments, sub_batch_size):
            sub_images = images[i:i+sub_batch_size]
            outputs = model(sub_images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            probs_visual_list.append(probs)
            
        probs_visual = np.concatenate(probs_visual_list, axis=0)  # Shape: [n_segments, 234]
        
        # Vectorized linear blend prior calculation
        final_probs = probs_visual * (1.0 - CFG.SPATIAL_BLEND_ALPHA) + probs_knn_prior * CFG.SPATIAL_BLEND_ALPHA
        
    # Append results
    for seg_idx in range(n_segments):
        end_time_sec = int((seg_idx + 1) * CFG.WINDOW_SECONDS)
        row_id = f"{filename}_{end_time_sec}"
        all_row_ids.append(row_id)
        all_predictions.append(final_probs[seg_idx])

print("Inference completed.")

# 5. Create final submission file
print("Formatting submission...")
submission_df = pd.DataFrame(all_predictions, columns=submission_labels)
submission_df.insert(0, 'row_id', all_row_ids)

submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} (Shape: {submission_df.shape})")

