import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv2d  # Assuming this is your residual conv implementation

class audio_only3(nn.Module):
    def __init__(self, input_size=187, hidden_size=256, num_layers=2, bidirectional=True):
        super(audio_only3, self).__init__()
        
        
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
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
    
    def forward(self, audio_sequences):
        """
        audio_sequences: (B, 1, 80, 16)
        """

        # Encode audio mel spectrogram with CNN
        audio_feat = self.audio_encoder(audio_sequences)  # (B, 512, H, W)
        audio_feat = F.adaptive_avg_pool2d(audio_feat, (1, 1))  # (B, 512, 1, 1)
        audio_feat = audio_feat.reshape(audio_feat.size(0), -1)  # (B, 512)
        audio_embedding = self.audio_proj(audio_feat)  # (B, 128)

        # Normalize for cosine similarity use
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)

        return audio_embedding
