import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from typing import Tuple
from src.config import Config
from src.data.transforms import Mixup
from src.engine.evaluator import calculate_macro_auc

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler,
    device: torch.device,
    mixup_fn: Mixup = None,
    logger = None,
    epoch: int = 0
) -> Tuple[float, float]:
    """
    Trains the model for one epoch.
    Integrates Mixup, Mixed Precision (AMP), Spatial Dropout, and Gradient Clipping.
    """
    model.train()
    
    total_loss = 0.0
    all_targets = []
    all_preds = []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (spectrograms, pe_coords, targets) in enumerate(progress_bar):
        spectrograms = spectrograms.to(device)
        pe_coords = pe_coords.to(device)
        targets = targets.to(device)
        
        # 1. Apply Mixup batch augmentation during training (if enabled)
        if mixup_fn is not None:
            spectrograms, targets = mixup_fn(spectrograms, targets)
            
        optimizer.zero_grad()
        
        # 2. Forward pass with PyTorch Auto-Mixed Precision (AMP)
        with torch.amp.autocast('cuda'):
            # Apply coordinate dropout during training
            outputs = model(spectrograms, pe_coords, apply_dropout=True)
            loss = criterion(outputs, targets)
            
        # Log batch loss for graphical plotting
        if logger is not None:
            logger.log_batch(epoch, batch_idx, loss.item())
            
        # 3. Backward pass with Gradient Scaling
        scaler.scale(loss).backward()
        
        # 4. Unscale gradients and clip norms to prevent gradient explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 5. Step optimizer and scaler
        scaler.step(optimizer)
        scaler.update()
        
        # Track statistics
        total_loss += loss.item()
        
        # Convert outputs to probabilities for epoch metric computation
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    avg_loss = total_loss / len(dataloader)
    
    # Calculate training macro AUC
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    epoch_auc = calculate_macro_auc(all_targets, all_preds)
    
    return avg_loss, epoch_auc

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation fold (OOF).
    Disables coordinate dropout and gradient calculations.
    """
    model.eval()
    
    total_loss = 0.0
    all_targets = []
    all_preds = []
    
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    
    for spectrograms, pe_coords, targets in progress_bar:
        spectrograms = spectrograms.to(device)
        pe_coords = pe_coords.to(device)
        targets = targets.to(device)
        
        # Run forward pass without coordinate dropout
        outputs = model(spectrograms, pe_coords, apply_dropout=False)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        
        # Apply sigmoid to outputs to get probabilities
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.cpu().numpy())
        
    avg_loss = total_loss / len(dataloader)
    
    # Calculate validation macro AUC
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    epoch_auc = calculate_macro_auc(all_targets, all_preds)
    
    return avg_loss, epoch_auc
