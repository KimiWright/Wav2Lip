import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv2d  # Assuming this is your residual conv implementation

class lmks_only3(nn.Module):
    def __init__(self, input_size=187, hidden_size=256, num_layers=2, bidirectional=True):
        super(lmks_only3, self).__init__()

        self.face_encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True  # input will be [B, T, D]
        )
        
        self.face_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
    
    def forward(self, face_sequences):
        """
        audio_sequences: (B, 1, 80, 16)
        face_sequences:  (B, 5, 187)
        """
        # Encode face landmarks with GRU
        face_output, _ = self.face_encoder(face_sequences)  # (B, 5, H*2)
        face_embedding = face_output.mean(dim=1)  # Mean pool over time: (B, H*2)
        face_embedding = self.face_proj(face_embedding)  # (B, 128)

        # Normalize for cosine similarity use
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return face_embedding
