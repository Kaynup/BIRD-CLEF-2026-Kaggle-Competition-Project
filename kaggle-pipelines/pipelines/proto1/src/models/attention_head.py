import torch
import torch.nn as nn
from src.config import Config

class MultiLabelAttentionHead(nn.Module):
    """
    Query2Label Multi-Label Attention Head.
    1. Holds 234 learnable species query embeddings.
    2. Performs Cross-Attention to query the vision backbone feature map.
    3. Performs Self-Attention over the 234 queries to model species co-occurrence.
    4. Projects each species embedding to a single logit.
    """
    def __init__(
        self, 
        num_classes: int = Config.NUM_CLASSES, 
        backbone_dim: int = Config.BACKBONE_OUT_DIM, 
        embed_dim: int = Config.SPECIES_EMB_DIM, 
        num_heads: int = Config.ATTENTION_HEADS
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # 1. Learnable Species Query Embeddings
        self.species_queries = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        
        # 2. Backbone Feature Projection
        self.feat_projection = nn.Conv2d(backbone_dim, embed_dim, kernel_size=1)
        
        # 3. Cross-Attention Layer (Queries attend to Audio Features)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)
        
        # 4. Self-Attention Layer (Queries attend to each other for Co-occurrence)
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(embed_dim)
        
        # 5. Shared Logit Projection Head
        # Projects [B, 234, embed_dim] -> [B, 234, 1] -> [B, 234]
        self.fc = nn.Linear(embed_dim, 1)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Input features shape: [Batch, backbone_dim, H_feat, W_feat]
        batch_size = features.size(0)
        
        # 1. Project and flatten features to [Batch, H*W, embed_dim]
        proj_feats = self.feat_projection(features)
        proj_feats = proj_feats.flatten(2).transpose(1, 2)  # Shape: [B, H*W, embed_dim]
        
        # 2. Expand species queries to match batch size: [Batch, 234, embed_dim]
        queries = self.species_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. Cross-Attention: Queries (Q) attend to Audio Features (K, V)
        attn_out, _ = self.cross_attn(query=queries, key=proj_feats, value=proj_feats)
        x = self.cross_norm(queries + attn_out)  # Residual and Norm
        
        # 4. Self-Attention: Species Queries attend to each other (Relational co-occurrence)
        self_attn_out, _ = self.self_attn(query=x, key=x, value=x)
        x = self.self_norm(x + self_attn_out)  # Residual and Norm
        
        # 5. Project each species token to a single logit
        logits = self.fc(x).squeeze(-1)  # Shape: [Batch, 234]
        
        return logits
