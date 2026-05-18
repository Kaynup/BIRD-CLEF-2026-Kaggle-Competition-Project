import torch
import torch.nn as nn
from typing import List
from src.config import Config

class SpatialMLP(nn.Module):
    """
    3-Layer Multilayer Perceptron mapping positional-encoded coordinates
    to species logit distributions. Includes coordinate dropout for regularization.
    """
    def __init__(self, layer_dims: List[int] = Config.SPATIAL_MLP_DIMS, dropout_prob: float = 0.2):
        super().__init__()
        
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_prob))
                
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, pe_coords: torch.Tensor, apply_dropout: bool = True) -> torch.Tensor:
        """
        Input Shape: [Batch, 36] (sinusoidal coordinate encodings)
        apply_dropout: If True (during training), applies coordinate dropout
                       replacing inputs with zeros to enforce audio backbone dependency.
        """
        if self.training and apply_dropout:
            # 1. Generate random binary mask [Batch, 1]
            mask = (torch.rand(pe_coords.size(0), 1, device=pe_coords.device) > Config.SPATIAL_DROPOUT_PROB).float()
            # 2. Apply spatial dropout
            pe_coords = pe_coords * mask
            
        spatial_logits = self.mlp(pe_coords)  # Shape: [Batch, 234]
        return spatial_logits
