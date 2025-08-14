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
class SyncNet_landmarks_attn(nn.Module):
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

        # Audio encoder: residual Conv2d blocks
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, audio_sequences, face_sequences):
        """
        audio_sequences: (B, 1, 80, 16)
        face_sequences:  (B, 5, 187)
        """
        # ---- Face landmarks: GRU + Attention pooling ----
        face_output, _ = self.face_encoder_rnn(face_sequences)  # (B, T, H*2)
        face_weighted, attn_weights = self.face_attention(face_output)  # (B, H*2)
        face_embedding = self.face_proj(face_weighted)  # (B, 128)

        # ---- Audio: CNN encoder + pooling ----
        audio_feat = self.audio_encoder(audio_sequences)  # (B, 512, H, W)
        audio_feat = F.adaptive_avg_pool2d(audio_feat, (1, 1))  # (B, 512, 1, 1)
        audio_feat = audio_feat.view(audio_feat.size(0), -1)  # (B, 512)
        audio_embedding = self.audio_proj(audio_feat)  # (B, 128)

        # ---- Normalize for cosine similarity ----
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)

        return audio_embedding, face_embedding #, attn_weights
