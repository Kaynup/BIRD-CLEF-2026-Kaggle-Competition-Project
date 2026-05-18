import torch
import os
from pathlib import Path
from src.config import Config
from src.utils.logging import ConsoleLogger

class GeneralizationCheckpointSaver:
    """
    Checkpoint manager that saves the model weights based on generalization.
    Generalization Score = Val AUC - gamma * max(0, Train AUC - Val AUC)
    Heavily penalizes models that overfit (high train AUC, low val AUC).
    """
    def __init__(self, output_dir: Path, fold: int, gamma: float = Config.GENERALIZATION_PENALTY_GAMMA):
        self.output_dir = output_dir
        self.fold = fold
        self.gamma = gamma
        self.best_gen_score = -float('inf')
        self.best_val_auc = 0.0
        
        self.best_path = output_dir / f"best_model_fold_{fold}.pth"
        self.last_path = output_dir / f"last_model_fold_{fold}.pth"
        
    def save(self, model: torch.nn.Module, epoch: int, train_auc: float, val_auc: float) -> bool:
        """
        Evaluates and saves model weights.
        Returns True if a new best generalization score was achieved and saved.
        """
        # Calculate generalization gap and score
        gap = max(0.0, train_auc - val_auc)
        gen_score = val_auc - self.gamma * gap
        
        # Save last model anyway
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_auc': train_auc,
            'val_auc': val_auc,
            'gen_score': gen_score
        }, self.last_path)
        
        # Check if new best generalization score is achieved
        is_best = False
        if gen_score > self.best_gen_score:
            self.best_gen_score = gen_score
            self.best_val_auc = val_auc
            is_best = True
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_auc': train_auc,
                'val_auc': val_auc,
                'gen_score': gen_score
            }, self.best_path)
            
            ConsoleLogger.log(
                f"[FOLD {self.fold}] NEW BEST Model Saved! "
                f"Epoch: {epoch:02d} | Gen Score: {gen_score:.4f} | Val AUC: {val_auc:.4f} (Gap: {gap:.4f})",
                level="SUCCESS"
            )
            
        return is_best
