import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNConformerVVAD(nn.Module):
    """
    VVAD model built on top of the LandmarkSTGCNConformer feature extractor.
    """
    def __init__(self, base_model, 
                 d_model=128, 
                 num_classes=1, 
                 dropout=0.3, 
                 attn_pool=False):
        """
        Args:
            base_model: An instance of LandmarkSTGCNConformer (feature extractor)
            d_model: Dimensionality of conformer output features
            num_classes: Number of VVAD output classes (1 for binary)
            dropout: Dropout rate before classification head
            attn_pool: If True, apply attention pooling over time
        """
        super().__init__()
        self.base_model = base_model
        self.attn_pool = attn_pool

        if attn_pool:
            self.attn = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        # VVAD classifier head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, x_rot = None, key_padding_mask=None):
        """
        Args:
            x: Tensor of shape (B, C, T, V)
        Returns:
            logits: (B, num_classes)
        """
        # Extract temporal features using your base ST-GCN + Conformer
        if x_rot is None:
            feat_seq = self.base_model(x, key_padding_mask=key_padding_mask)
        else:
            feat_seq = self.base_model(x, x_rot, key_padding_mask=key_padding_mask)
        # Expected shape: (B, T, D)

        # Temporal pooling
        if self.attn_pool:
            # Attention-based temporal pooling
            w = self.attn(feat_seq)          # (B, T, 1)
            w = torch.softmax(w, dim=1)
            pooled = (feat_seq * w).sum(dim=1)  # (B, D)
        else:
            # Mean pooling across time
            pooled = feat_seq.mean(dim=1)    # (B, D)

        # VVAD classification
        logits = self.fc(pooled)             # (B, num_classes)
        return logits