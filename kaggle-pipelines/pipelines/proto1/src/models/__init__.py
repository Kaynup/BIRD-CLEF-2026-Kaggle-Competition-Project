import torch
import torch.nn as nn
from src.config import Config
from src.models.backbone import AudioVisionBackbone
from src.models.spatial_mlp import SpatialMLP
from src.models.attention_head import MultiLabelAttentionHead
from src.models.fusion import LogitGatedFusion

class MultiModalBirdModel(nn.Module):
    """
    Unified Multi-Modal Model assembling all designed modules:
    1. Vision Backbone (extracts spatial-temporal features from Spectrogram).
    2. Spatial MLP (projects coordinate encodings to regional spatial logits).
    3. Multi-Label Attention Head (Query2Label cross & self attention).
    4. Logit Gated Fusion (learnable gate to balance audio and space logits).
    """
    def __init__(
        self,
        backbone_name: str = Config.BACKBONE_NAME,
        pretrained: bool = True,
        num_classes: int = Config.NUM_CLASSES
    ):
        super().__init__()
        
        # 1. Vision Backbone
        self.backbone = AudioVisionBackbone(backbone_name=backbone_name, pretrained=pretrained)
        
        # 2. Relational Species Attention Head
        self.attention_head = MultiLabelAttentionHead(
            num_classes=num_classes,
            backbone_dim=self.backbone.out_channels,
            embed_dim=Config.SPECIES_EMB_DIM,
            num_heads=Config.ATTENTION_HEADS
        )
        
        # 3. Spatial MLP
        self.spatial_mlp = SpatialMLP(
            layer_dims=Config.SPATIAL_MLP_DIMS,
            dropout_prob=0.2
        )
        
        # 4. Gated Fusion Layer
        self.fusion = LogitGatedFusion()
        
    def forward(
        self, 
        spectrogram: torch.Tensor, 
        pe_coords: torch.Tensor, 
        apply_dropout: bool = True
    ) -> torch.Tensor:
        """
        spectrogram: [Batch, 3, N_MELS, TIME]
        pe_coords: [Batch, 36]
        """
        # 1. Extract visual features
        features = self.backbone(spectrogram)  # [B, channels, H, W]
        
        # 2. Query features using Relational Attention Head
        logits_audio = self.attention_head(features)  # [B, 234]
        
        # 3. Compute spatial logits dynamically
        logits_spatial = self.spatial_mlp(pe_coords, apply_dropout=apply_dropout)  # [B, 234]
        
        # 4. Fuse audio and spatial logits
        logits_final = self.fusion(logits_audio, logits_spatial)  # [B, 234]
        
        return logits_final
