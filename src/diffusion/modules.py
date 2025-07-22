import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Module for sinusoidal position embeddings to encode the timestep 't'."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """A convolutional block with time embeddings and residual connections."""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        
        # Residual connection if channels are different
        self.residual_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.norm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        h = h + time_emb.unsqueeze(-1)
        h = self.norm2(self.relu(self.conv2(h)))
        return h + self.residual_conv(x)

class Downsample(nn.Module):
    """Downsamples the signal length using a strided convolution."""
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)

class Upsample(nn.Module):
    """Upsamples the signal length using a transpose convolution."""
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.up(x)

class UNet_1D(nn.Module):
    """A 1D U-Net architecture with a clear, standard structure."""
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=32):
        super().__init__()
        
        # --- Time Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # --- Encoder (Downsampling Path) ---
        self.initial_conv = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList([
            Block(64, 128, time_emb_dim),
            Block(128, 256, time_emb_dim),
            Block(256, 512, time_emb_dim)
        ])
        self.down_samplers = nn.ModuleList([
            Downsample(128),
            Downsample(256),
            Downsample(512)
        ])

        # --- Bottleneck ---
        self.mid_block1 = Block(512, 1024, time_emb_dim)
        self.mid_block2 = Block(1024, 1024, time_emb_dim)

        # --- Decoder (Upsampling Path) ---
        self.up_blocks = nn.ModuleList([
            Block(1024 + 512, 512, time_emb_dim),
            Block(512  + 256, 256, time_emb_dim),
            Block(256  + 128,  64, time_emb_dim)
        ])
        self.up_samplers = nn.ModuleList([
            Upsample(1024),
            Upsample(512),
            Upsample(256)
        ])

        # --- Final Output ---
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.initial_conv(x)
        
        residuals = []
        # Encoder
        for block, downsampler in zip(self.down_blocks, self.down_samplers):
            x = block(x, t)
            residuals.append(x)
            x = downsampler(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # Decoder
        for block, upsampler, res in zip(self.up_blocks, self.up_samplers, reversed(residuals)):
            x = upsampler(x)
            x = torch.cat((x, res), dim=1)
            x = block(x, t)
            
        return self.final_conv(x) 