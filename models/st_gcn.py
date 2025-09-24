import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utility: sinusoidal positions
# ---------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# ---------------------------
# ST-GCN building blocks
# ---------------------------
class STGCNBlock(nn.Module):
    """
    Spatial-temporal GCN block.
    Input/Output shape: [B, C, T, V]
    A: [K, V, V] adjacency partitions (registered buffer).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        A: Optional[torch.Tensor] = None,
        temporal_kernel_size: int = 9,
        stride_t: int = 1,
        residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert temporal_kernel_size % 2 == 1, "temporal_kernel_size should be odd"

        if A is None:
            raise ValueError("Adjacency A must be provided as [K, V, V].")

        self.K, self.V, _ = A.shape
        self.register_buffer("A", A)

        self.spatial_conv = nn.Conv2d(in_channels * self.K, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        pad = (temporal_kernel_size // 2, 0)
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(temporal_kernel_size, 1),
            padding=pad,
            stride=(stride_t, 1),
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU(inplace=True)
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride_t == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride_t, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T, V]
        B, C, T, V = x.shape
        assert V == self.V, f"Expected V={self.V}, got {V}"

        # Spatial graph conv: concat across K partitions
        # For each A_k: X_k = X @ A_k (over V dimension)
        xs = []
        for k in range(self.K):
            Ak = self.A[k]  # [V, V]
            xk = torch.einsum("nctv,vw->nctw", x, Ak)  # [B, C, T, V]
            xs.append(xk)
        x_cat = torch.cat(xs, dim=1)  # [B, C*K, T, V]

        # 1x1 conv to mix channels across partitions
        y = self.spatial_conv(x_cat)          # [B, C_out, T, V]
        y = self.bn1(y)
        y = self.act(y)

        # Temporal conv
        y = self.temporal_conv(y)             # [B, C_out, T, V]
        y = self.bn2(y)

        # Residual and activation
        y = y + self.residual(x)
        y = self.act(y)
        y = self.do(y)
        return y


class STGCNFrontEnd(nn.Module):
    """
    6 ST-GCN blocks.
    First block in_channels=2, all blocks out_channels=64.
    """
    def __init__(
        self,
        num_nodes: int,
        A: torch.Tensor,              # [K, V, V]
        temporal_kernel_size: int = 9,
        dropout: float = 0.0
    ):
        super().__init__()
        assert A.dim() == 3, "A must be [K, V, V]"
        K, V, _ = A.shape
        assert V == num_nodes, "Adjacency V must equal num_nodes"

        blocks = []
        in_ch = 2     # per spec: first block has input of 2 channels (x,y)
        out_ch = 64

        # Block 1
        blocks.append(
            STGCNBlock(
                in_channels=in_ch, out_channels=out_ch,
                A=A, temporal_kernel_size=temporal_kernel_size, stride_t=1, dropout=dropout
            )
        )
        # Blocks 2..6 (all 64 -> 64)
        for _ in range(5):
            blocks.append(
                STGCNBlock(
                    in_channels=out_ch, out_channels=out_ch,
                    A=A, temporal_kernel_size=temporal_kernel_size, stride_t=1, dropout=dropout
                )
            )

        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 2, T, V]  (channels = 2: x,y)
           If you have input as [B, T, V, 2], call x = x.permute(0, 3, 1, 2) first.
        returns: [B, 64, T, V]
        """
        return self.net(x)


# ---------------------------
# Conformer (temporal backend)
# ---------------------------
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)


class ConformerConvModule(nn.Module):
    """
    Conformer conv module:
    LN -> 1x1 PW conv (expand) -> GLU -> depthwise conv -> BN -> Swish -> 1x1 PW conv (project)
    """
    def __init__(self, d_model: int, expansion: int = 2, kernel_size: int = 31):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"

        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, d_model * expansion * 2, kernel_size=1)  # *2 for GLU split
        self.dw = nn.Conv1d(d_model * expansion, d_model * expansion, kernel_size=kernel_size,
                            padding=kernel_size // 2, groups=d_model * expansion)
        self.bn = nn.BatchNorm1d(d_model * expansion)
        self.swish = Swish()
        self.pw2 = nn.Conv1d(d_model * expansion, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        y = self.ln(x)
        y = y.transpose(1, 2)  # [B, D, T]
        y = self.pw1(y)        # [B, 2*E*D, T]
        a, b = torch.chunk(y, 2, dim=1)
        y = a * torch.sigmoid(b)  # GLU, [B, E*D, T]
        y = self.dw(y)
        y = self.bn(y)
        y = self.swish(y)
        y = self.pw2(y)        # [B, D, T]
        y = y.transpose(1, 2)  # [B, T, D]
        return y


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.ln(x)
        return self.ff(y)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, conv_kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mha_ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.conv = ConformerConvModule(d_model, expansion=2, kernel_size=conv_kernel_size)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Macaron feed-forward (scaled 1/2)
        x = x + 0.5 * self.ff1(x)

        # MHSA
        y = self.mha_ln(x)
        attn_out, _ = self.mha(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out

        # Conv module
        x = x + self.conv(x)

        # Second feed-forward (scaled 1/2)
        x = x + 0.5 * self.ff2(x)

        # Final layer norm
        return self.final_ln(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 256,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.pe = SinusoidalPositionalEncoding(d_model) if use_positional_encoding else nn.Identity()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, d_ff, conv_kernel_size, dropout) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, D]
        key_padding_mask: [B, T] with True for padded timesteps (optional)
        """
        x = self.pe(x) if isinstance(self.pe, SinusoidalPositionalEncoding) else x
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x  # [B, T, D]


# ---------------------------
# Full Model: ST-GCN -> Linear(s)+Mish -> Conformer
# ---------------------------
class LandmarkSTGCNConformer(nn.Module):
    """
    Front-end: 6 ST-GCN blocks (first block in_channels=2, all blocks out_channels=64).
    Then: Linear layers + Mish to produce landmark feature vectors per time step.
    Back-end: Conformer encoder over time.

    Expected input: [B, 2, T, V] (channels = x,y coords per landmark)
    Returns: temporal features [B, T, D] after conformer.
    """
    def __init__(
        self,
        num_nodes: int,
        A: torch.Tensor,                # [K, V, V] adjacency (normalized or raw)
        d_model: int = 128,
        post_linear_hidden: int = 128,  # hidden size in the linear head before conformer
        temporal_kernel_size: int = 9,
        stgcn_dropout: float = 0.0,
        conformer_layers: int = 4,
        conformer_heads: int = 4,
        conformer_ff: int = 256,
        conformer_conv_kernel: int = 31,
        conformer_dropout: float = 0.1,
    ):
        super().__init__()

        # ST-GCN front-end
        self.stgcn = STGCNFrontEnd(
            num_nodes=num_nodes,
            A=A,
            temporal_kernel_size=temporal_kernel_size,
            dropout=stgcn_dropout
        )

        # Pool across nodes -> per-time features
        self.node_pool = nn.AdaptiveAvgPool2d((None, 1))  # keep T, pool V -> 1

        # Linear layers + Mish (per spec)
        # Input channels after pooling: 64
        self.post_linear = nn.Sequential(
            nn.Linear(64, post_linear_hidden, bias=True),
            nn.Mish(),
            nn.Linear(post_linear_hidden, d_model, bias=True),
            nn.Mish(),  # final Mish activation that outputs the extracted landmark features
        )

        # Temporal Conformer back-end
        self.conformer = ConformerEncoder(
            d_model=d_model,
            n_layers=conformer_layers,
            n_heads=conformer_heads,
            d_ff=conformer_ff,
            conv_kernel_size=conformer_conv_kernel,
            dropout=conformer_dropout,
            use_positional_encoding=True,
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, 2, T, V]  (if you have [B, T, V, 2], permute to this)
        key_padding_mask: optional [B, T] mask (True for padding)
        returns: [B, T, D]
        """
        # ST-GCN front-end
        y = self.stgcn(x)  # [B, 64, T, V]

        # Pool across nodes (V)
        y = self.node_pool(y)           # [B, 64, T, 1]
        y = y.squeeze(-1).transpose(1, 2)  # [B, T, 64]

        # Linear layers + Mish -> landmark features
        feats = self.post_linear(y)     # [B, T, D]

        # Temporal Conformer
        out = the_features = self.conformer(feats, key_padding_mask=key_padding_mask)  # [B, T, D]
        return out

class LandmarkSTGCNConformerWithOrientation(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        A: torch.Tensor,                # [K, V, V] adjacency
        d_model: int = 128,
        post_linear_hidden: int = 128,  # hidden size before conformer
        temporal_kernel_size: int = 9,
        stgcn_dropout: float = 0.0,
        conformer_layers: int = 4,
        conformer_heads: int = 4,
        conformer_ff: int = 256,
        conformer_conv_kernel: int = 31,
        conformer_dropout: float = 0.1,
    ):
        super().__init__()

        # ST-GCN front-end
        self.stgcn = STGCNFrontEnd(
            num_nodes=num_nodes,
            A=A,
            temporal_kernel_size=temporal_kernel_size,
            dropout=stgcn_dropout,
        )

        # Pool across nodes (average landmarks → one vector per frame)
        self.node_pool = nn.AdaptiveAvgPool2d((None, 1))

        # Linear layers + Mish, now input dim is 64 (landmarks) + 3 (roll,pitch,yaw) = 67
        self.post_linear = nn.Sequential(
            nn.Linear(64 + 3, post_linear_hidden, bias=True),
            nn.Mish(),
            nn.Linear(post_linear_hidden, d_model, bias=True),
            nn.Mish(),
        )

        # Temporal Conformer back-end
        self.conformer = ConformerEncoder(
            d_model=d_model,
            n_layers=conformer_layers,
            n_heads=conformer_heads,
            d_ff=conformer_ff,
            conv_kernel_size=conformer_conv_kernel,
            dropout=conformer_dropout,
            use_positional_encoding=True,
        )

    def forward(self, x: torch.Tensor, orientation: torch.Tensor, key_padding_mask=None):
        """
        Args:
            x: [B, 2, T, V] landmark coords
            orientation: [B, T, 3] roll, pitch, yaw per frame
            key_padding_mask: optional [B, T] mask for padded timesteps
        Returns:
            [B, T, D] features
        """
        # Landmark features from ST-GCN
        y = self.stgcn(x)                       # [B, 64, T, V]
        y = self.node_pool(y).squeeze(-1)       # [B, 64, T]
        y = y.transpose(1, 2)                   # [B, T, 64]

        # Fuse orientation
        feats = torch.cat([y, orientation], dim=-1)  # [B, T, 67]

        # Linear + Mish projection
        feats = self.post_linear(feats)              # [B, T, D]

        # Temporal Conformer
        out = self.conformer(feats, key_padding_mask=key_padding_mask)  # [B, T, D]
        return out


# ---------------------------
# Helper to build adjacency
# ---------------------------
from typing import List, Tuple
def build_adjacency(num_nodes: int, edges: List[Tuple[int, int]], num_partitions: int = 1) -> torch.Tensor:
    """
    Simple helper to construct an adjacency tensor A with K partitions.
    Here we put the same adjacency into each partition by default.
    - num_nodes: V
    - edges: list of (i, j) undirected edges
    returns A: [K, V, V]
    """
    A = torch.eye(num_nodes, dtype=torch.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Normalize by degree (symmetric)
    D = torch.diag(torch.pow(A.sum(-1).clamp(min=1.0), -0.5))
    A_norm = D @ A @ D
    return A_norm.unsqueeze(0).repeat(num_partitions, 1, 1)  # [K, V, V]


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    V = 68                 # number of landmarks (example)
    K = 1                  # partitions
    # Dummy chain graph (or plug in your real edges)
    edges = [(i, i+1) for i in range(V-1)]
    A = build_adjacency(V, edges, num_partitions=K)  # [K, V, V]

    model = LandmarkSTGCNConformer(
        num_nodes=V,
        A=A,
        d_model=128,
        post_linear_hidden=128,
        conformer_layers=4,
        conformer_heads=4,
        conformer_ff=256,
        conformer_conv_kernel=31
    )

    B, T = 2, 120
    x = torch.randn(B, 2, T, V)  # [B, 2, T, V]
    y = model(x)                 # [B, T, 128]
    print(y.shape)
