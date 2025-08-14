import torch
from torch import nn
import torch.nn.functional as F
from .conv import Conv2d  # Assuming your residual conv implementation is here

# --------------------
# Attention Pooling
# --------------------
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, attention_dim=64):
        super().__init__()
        self.att_proj = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)  # unnormalized attention score per frame
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        attn_scores = self.att_proj(x)  # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        weighted_sum = (x * attn_weights).sum(dim=1)  # (B, D)
        return weighted_sum, attn_weights
    
# --------------------
# Main Model
# --------------------
class lmks_only_attn(nn.Module):
    def __init__(self, input_size=187, hidden_size=256, num_layers=2, bidirectional=True):
        super().__init__()

        # Face encoder: GRU + Attention Pooling
        self.face_encoder_rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.face_attention = AttentionPooling(hidden_size * 2)
        self.face_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, face_sequences):
        """
        face_sequences:  (B, 5, 187)
        """
        # ---- Face landmarks: GRU + Attention pooling ----
        face_output, _ = self.face_encoder_rnn(face_sequences)  # (B, T, H*2)
        face_weighted, attn_weights = self.face_attention(face_output)  # (B, H*2)
        face_embedding = self.face_proj(face_weighted)  # (B, 128)

        # ---- Normalize for cosine similarity ----
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return face_embedding #, attn_weights