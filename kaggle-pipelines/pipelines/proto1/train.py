import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from src.config import Config
from src.utils.logging import ConsoleLogger, MetricLogger
from src.utils.spatial import compute_pantanal_hotness_vector
from src.data.transforms import SpecAugment, Mixup
from src.data.dataset import BirdCLEFDataset
from src.models import MultiModalBirdModel
from src.engine.trainer import train_one_epoch, validate
from src.engine.checkpoint import GeneralizationCheckpointSaver

def main():
    ConsoleLogger.log("Starting BirdCLEF 2026 Multi-Modal Pipeline Training...")
    
    # 1. Allocate next run directory dynamically
    base_dir = Config.PIPELINE_ROOT / "outputs"
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith("run-"):
            try:
                num = int(p.name.split("-")[1])
                existing_runs.append(num)
            except ValueError:
                pass
    next_num = max(existing_runs) + 1 if existing_runs else 1
    run_dir = base_dir / f"run-{next_num}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the dynamic Config.OUTPUT_DIR to this run directory
    Config.OUTPUT_DIR = run_dir
    ConsoleLogger.log(f"All model weights, histories, and plots will save to: {Config.OUTPUT_DIR}", level="SUCCESS")
    
    # 2. Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ConsoleLogger.log(f"Active Training Device: {device}")
    if device.type == "cuda":
        ConsoleLogger.log(f"Device Name: {torch.cuda.get_device_name(0)}")
        # Enable benchmark mode for faster convolution operations
        torch.backends.cudnn.benchmark = True

    # 2. Load Datasets & Target Species Mapping
    sub_df = pd.read_csv(Config.SAMPLE_SUB_CSV)
    target_species = list(sub_df.columns)[1:]  # Exclude row_id
    species_to_idx = {species: idx for idx, species in enumerate(target_species)}
    ConsoleLogger.log(f"Loaded {len(target_species)} target species from sample submission columns.")

    train_df = pd.read_csv(Config.TRAIN_CSV)
    soundscapes_df = pd.read_csv(Config.SOUNDSCAPES_CSV)
    ConsoleLogger.log(f"train.csv shape: {train_df.shape} | train_soundscapes_labels.csv shape: {soundscapes_df.shape}")

    # 3. Stratified Group K-Fold split on labeled soundscapes
    # We group by 'filename' to ensure complete segment isolation and zero validation leakage!
    sgkf = StratifiedGroupKFold(n_splits=Config.NUM_FOLDS)
    
    # We will stratify based on the presence of any label to balance splits
    soundscapes_df['stratify_label'] = soundscapes_df['primary_label'].apply(lambda x: str(x).split(';')[0])
    
    folds = list(sgkf.split(
        soundscapes_df, 
        soundscapes_df['stratify_label'], 
        groups=soundscapes_df['filename']
    ))
    
    # Pre-compute and save Pantanal Hotness Prior Vector
    ConsoleLogger.log("Computing Pantanal Hotness Vector...")
    hotness_vector = compute_pantanal_hotness_vector(
        Config.TRAIN_CSV, Config.TAXONOMY_CSV, Config.SAMPLE_SUB_CSV
    ).to(device)
    
    # We will train Fold 0 first as our primary Prototype 1 pipeline
    for fold in range(1):
        ConsoleLogger.log(f"=== Starting Fold {fold} Training ===")
        train_idx, val_idx = folds[fold]
        
        fold_train_soundscapes = soundscapes_df.iloc[train_idx].copy()
        fold_val_soundscapes = soundscapes_df.iloc[val_idx].copy()
        
        # Training set = ALL short audio clips (XC/iNat) + 4/5ths of the labeled soundscapes
        combined_train_df = pd.concat([train_df, fold_train_soundscapes], ignore_index=True)
        # Validation set = OOF 1/5th labeled soundscapes (zero leakage target)
        combined_val_df = fold_val_soundscapes
        
        ConsoleLogger.log(
            f"[Fold {fold}] Train Dataset size: {len(combined_train_df)} "
            f"({len(train_df)} short clips + {len(fold_train_soundscapes)} soundscape windows) | "
            f"Validation Dataset size: {len(combined_val_df)} soundscape windows"
        )
        
        # 4. Create Datasets and DataLoaders
        spec_augment = SpecAugment(freq_mask_param=Config.FREQ_MASK_PARAM, time_mask_param=Config.TIME_MASK_PARAM)
        
        train_dataset = BirdCLEFDataset(
            df=combined_train_df, 
            species_to_idx=species_to_idx, 
            is_train=True, 
            spec_augment=spec_augment
        )
        
        val_dataset = BirdCLEFDataset(
            df=combined_val_df, 
            species_to_idx=species_to_idx, 
            is_train=False
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
        
        # 5. Instantiate Model and Optimizer
        model = MultiModalBirdModel(
            backbone_name=Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=Config.NUM_CLASSES
        ).to(device)
        
        # Loss function (Binary Cross Entropy for Multi-Label target probabilities)
        criterion = nn.BCEWithLogitsLoss()
        
        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Cosine Annealing Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=Config.EPOCHS, 
            eta_min=Config.COSINE_LR_MIN
        )
        
        # Automatic Mixed Precision (AMP) gradient scaler
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler(device.type)
        else:
            scaler = torch.cuda.amp.GradScaler()
        
        # Augmentation (Batch level multi-label Mixup)
        mixup_fn = Mixup(alpha=Config.MIXUP_ALPHA, prob=Config.MIXUP_PROB)
        
        # Loggers & Checkpoint Saver
        logger = MetricLogger(Config.OUTPUT_DIR, fold)
        saver = GeneralizationCheckpointSaver(Config.OUTPUT_DIR, fold)
        
        # 6. Training & Validation Loop
        for epoch in range(1, Config.EPOCHS + 1):
            # Train one epoch
            train_loss, train_auc = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                device=device,
                mixup_fn=mixup_fn,
                logger=logger,
                epoch=epoch
            )
            
            # Validate OOF
            val_loss, val_auc = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device
            )
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Compute generalization score
            gap = max(0.0, train_auc - val_auc)
            gen_score = val_auc - Config.GENERALIZATION_PENALTY_GAMMA * gap
            
            # Log metric records
            logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_auc=train_auc,
                val_loss=val_loss,
                val_auc=val_auc,
                gen_score=gen_score,
                lr=current_lr
            )
            
            # Save Checkpoints
            is_best = saver.save(model, epoch, train_auc, val_auc)
            
            # Step Scheduler
            scheduler.step()
            
        ConsoleLogger.log(f"Fold {fold} Completed. Best Generalization Score: {saver.best_gen_score:.4f} | Best Val AUC: {saver.best_val_auc:.4f}")

if __name__ == "__main__":
    main()
