import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d

class SyncNet_landmarks2(nn.Module):
    def __init__(self):
        super(SyncNet_landmarks2, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(5, 32, kernel_size=(1, 7), stride=1, padding=(0, 3)),

            Conv2d(32, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), residual=True),
            Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), residual=True),

            Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            Conv2d(128, 128, kernel_size=(1, 3), stride=1, padding=(0, 1), residual=True),
            Conv2d(128, 128, kernel_size=(1, 3), stride=1, padding=(0, 1), residual=True),
            Conv2d(128, 128, kernel_size=(1, 3), stride=1, padding=(0, 1), residual=True),

            Conv2d(128, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            Conv2d(256, 256, kernel_size=(1, 3), stride=1, padding=(0, 1), residual=True),
            Conv2d(256, 256, kernel_size=(1, 3), stride=1, padding=(0, 1), residual=True),

            Conv2d(256, 512, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            Conv2d(512, 512, kernel_size=(1, 3), stride=1, padding=(0, 1), residual=True),
            Conv2d(512, 512, kernel_size=(1, 3), stride=1, padding=(0, 1), residual=True),

            Conv2d(512, 512, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            Conv2d(512, 512, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            Conv2d(512, 512, kernel_size=(1, 1), stride=1, padding=0),

            # --- collapse spatial dimensions to 1×1 ---
            nn.AdaptiveAvgPool2d((1, 1)),   # → [64, 512, 1, 1]
            nn.Flatten(1),                  # → [64, 512]
        )

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
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences.unsqueeze(2))
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding
