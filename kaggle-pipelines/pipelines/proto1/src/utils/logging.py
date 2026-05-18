import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

# Prevent X11 GUI server crashes in headless training systems
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ConsoleLogger:
    @staticmethod
    def log(message: str, level: str = "INFO"):
        # Color formatting codes for professional CLI reporting
        colors = {
            "INFO": "\033[94m",    # Blue
            "SUCCESS": "\033[92m", # Green
            "WARNING": "\033[93m", # Yellow
            "ERROR": "\033[91m",   # Red
            "RESET": "\033[0m"
        }
        c_start = colors.get(level, colors["INFO"])
        c_reset = colors["RESET"]
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {c_start}[{level}]{c_reset} {message}")

class MetricLogger:
    def __init__(self, log_dir: Path, fold: int):
        self.log_dir = log_dir
        self.fold = fold
        self.log_file = log_dir / f"fold_{fold}_history.csv"
        self.batch_log_file = log_dir / f"fold_{fold}_batch_history.csv"
        
        self.history = []
        self.batch_history = []
        
    def log_batch(self, epoch: int, batch: int, loss: float):
        record = {
            "epoch": epoch,
            "batch": batch,
            "train_loss": loss,
            "timestamp": datetime.now().isoformat()
        }
        self.batch_history.append(record)
        
        # Save batch history periodically (e.g., every 50 batches to save disk I/O)
        if len(self.batch_history) % 50 == 0:
            df = pd.DataFrame(self.batch_history)
            df.to_csv(self.batch_log_file, index=False)
            
    def log_epoch(self, epoch: int, train_loss: float, train_auc: float, val_loss: float, val_auc: float, gen_score: float, lr: float):
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "generalization_score": gen_score,
            "lr": lr,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(record)
        
        # Save epoch history to CSV
        df = pd.DataFrame(self.history)
        df.to_csv(self.log_file, index=False)
        
        # Also flush any remaining batch history to disk
        df_batch = pd.DataFrame(self.batch_history)
        df_batch.to_csv(self.batch_log_file, index=False)
        
        # Print a beautiful summary table
        ConsoleLogger.log(
            f"Epoch {epoch:02d} Summary: "
            f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} || "
            f"Gen Score: {gen_score:.4f} | LR: {lr:.2e}",
            level="INFO"
        )
        
        # Generate diagnostic plots
        try:
            self.plot_metrics()
        except Exception as e:
            ConsoleLogger.log(f"Failed to generate plots: {str(e)}", level="WARNING")
            
    def plot_metrics(self):
        """
        Generates and saves premium training diagnostic plots.
        1. fold_X_batch_losses.png (shows batch losses with moving average).
        2. fold_X_epoch_metrics.png (shows loss, AUC, and Generalization Score).
        """
        # --- Plot 1: Batch-Level Losses ---
        if len(self.batch_history) > 0:
            df_batch = pd.DataFrame(self.batch_history)
            plt.figure(figsize=(10, 5))
            
            # Plot raw noisy batch loss in light blue
            plt.plot(df_batch["train_loss"], color="#A0C4FF", alpha=0.5, label="Raw Batch Loss")
            
            # Plot smoothed moving average (window=15 batches) in solid royal blue
            if len(df_batch) > 15:
                smooth_loss = df_batch["train_loss"].rolling(window=15, min_periods=1).mean()
                plt.plot(smooth_loss, color="#005F73", linewidth=2.0, label="Smoothed Loss (15-step MA)")
                
            plt.title(f"Fold {self.fold} - Batch Training Loss History", fontsize=12, fontweight="bold", pad=15)
            plt.xlabel("Training Step (Batch)", fontsize=10)
            plt.ylabel("BCE Loss", fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(frameon=True, facecolor="white", edgecolor="none")
            plt.tight_layout()
            
            plt.savefig(self.log_dir / f"fold_{self.fold}_batch_losses.png", dpi=150)
            plt.close()
            
        # --- Plot 2: Epoch-Level Metrics (Loss, AUC & Generalization Score) ---
        if len(self.history) > 0:
            df_epoch = pd.DataFrame(self.history)
            epochs = df_epoch["epoch"]
            
            # Find the best epoch based on maximum Generalization Score
            best_idx = df_epoch["generalization_score"].idxmax()
            best_epoch = df_epoch.loc[best_idx, "epoch"]
            best_gen_score = df_epoch.loc[best_idx, "generalization_score"]
            best_val_auc = df_epoch.loc[best_idx, "val_auc"]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
            
            # Subplot 1: Train Loss vs Val Loss
            ax1.plot(epochs, df_epoch["train_loss"], marker="o", color="#3D348B", linewidth=2, label="Train Loss")
            ax1.plot(epochs, df_epoch["val_loss"], marker="s", color="#F7B267", linewidth=2, label="Val Loss")
            ax1.axvline(x=best_epoch, color="#E76F51", linestyle="--", alpha=0.7, label=f"Best Checkpoint (Ep {best_epoch})")
            ax1.set_title(f"Fold {self.fold} - Training & Validation Loss", fontsize=12, fontweight="bold", pad=10)
            ax1.set_ylabel("BCE Loss", fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(frameon=True, facecolor="white")
            
            # Subplot 2: Train AUC, Val AUC, and Generalization Score
            ax2.plot(epochs, df_epoch["train_auc"], marker="o", color="#2A9D8F", linewidth=2, label="Train AUC")
            ax2.plot(epochs, df_epoch["val_auc"], marker="s", color="#457B9D", linewidth=2, label="Val AUC")
            ax2.plot(epochs, df_epoch["generalization_score"], marker="^", color="#E76F51", linewidth=2.5, linestyle="-.", label="Generalization Score")
            
            # Highlight best epoch with vertical line and star marker
            ax2.axvline(x=best_epoch, color="#E76F51", linestyle="--", alpha=0.7)
            ax2.scatter(best_epoch, best_gen_score, color="#D62828", marker="*", s=250, zorder=5, label=f"Best Checkpoint (Score: {best_gen_score:.4f})")
            
            ax2.set_title(f"Fold {self.fold} - Model Generalization Metrics", fontsize=12, fontweight="bold", pad=10)
            ax2.set_xlabel("Epoch", fontsize=10)
            ax2.set_ylabel("Macro ROC-AUC / Generalization", fontsize=10)
            ax2.set_xticks(epochs)
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(frameon=True, facecolor="white", loc="lower left")
            
            plt.tight_layout()
            plt.savefig(self.log_dir / f"fold_{self.fold}_epoch_metrics.png", dpi=150)
            plt.close()

