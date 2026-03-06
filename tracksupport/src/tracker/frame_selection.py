from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class SimpleFrameSelector(nn.Module):
    def __init__(self, num_frames: int, frame_interval: Tuple[int, int], num_queries: int = 1, num_probes: int = 4, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.num_queries = num_queries
        self.num_probes = num_probes
        self.embed_dim = embed_dim
        
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, stride=2), 
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.scale = embed_dim ** -0.5

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        B, V, T, C, H, W = videos.shape
        assert T >= self.num_frames, f"Number of frames {T} is less than required {self.num_frames}"

        # 1. Select Probes
        probe_idxs = torch.linspace(0, T-1, self.num_probes, device=videos.device).long()
        probes = videos[:, :, probe_idxs] # (B, V, num_probes, C, H, W)

        # 2. Encode Probes
        flat_probes = probes.view(B * V * self.num_probes, C, H, W)
        encoded_probes = self.input_proj(flat_probes) # (B*V*num_probes, D)
        encoded_probes = encoded_probes.view(B * V, self.num_probes, self.embed_dim)

        # 3. Cross Attention
        queries = self.query_embed.expand(B * V, -1, -1)
        
        # (B*V, Q, D) x (B*V, D, num_probes) -> (B*V, Q, num_probes)
        attn_logits = torch.bmm(queries, encoded_probes.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1) # (B*V, Q, num_probes)

        # 4. Interpolate to T frames
        interpolated_att = F.interpolate(attn_weights, size=T, mode='linear', align_corners=True) # (B*V, Q, T)
        
        # Normalize
        probs = interpolated_att / (interpolated_att.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Aggregate queries
        frame_probs = probs.mean(dim=1) # (B*V, T)

        # 5. Sample
        selected_indices = torch.multinomial(frame_probs, self.num_frames, replacement=False)
        selected_indices, _ = torch.sort(selected_indices, dim=-1)

        # 6. Gather
        flat_videos = videos.view(B * V, T, C, H, W)
        row_indices = torch.arange(B * V, device=videos.device).unsqueeze(-1)
        selected_frames = flat_videos[row_indices, selected_indices] # (B*V, num_frames, C, H, W)

        return selected_frames.view(B, V, self.num_frames, C, H, W)


        