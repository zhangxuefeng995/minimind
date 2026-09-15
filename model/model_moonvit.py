"""MoonViT-V2 的 MiniMind 规模视觉塔。

K3 原文：约 0.4B、27 层、无 bias、RMSNorm，从零用 next-token prediction 和 LLM 一起训；
视频先做空间注意力再做时间注意力，最后 2×2 pixel-shuffle 接到 LLM。

这里只保留同一条数据通路和模块形状，层数/宽度跟随 MiniMind，不复现 0.4B 视觉塔。
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


class MoonViTAttention(nn.Module):
    """无 bias 的多头注意力；空间/时间两条通路复用同一套投影。"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError('vision hidden_size 必须能被 num_heads 整除')
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C]
        bsz, seq_len, hidden = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        out = attn.transpose(1, 2).reshape(bsz, seq_len, hidden)
        return self.proj(out)


class MoonViTMlp(nn.Module):
    """无 bias 的 SwiGLU-style MLP（gate + up），与 LLM 侧风格接近。"""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        inner = 64 * ((inner + 63) // 64)
        self.gate_proj = nn.Linear(hidden_size, inner, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoonViTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, eps: float, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=eps)
        self.attn = MoonViTAttention(hidden_size, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(hidden_size, eps=eps)
        self.mlp = MoonViTMlp(hidden_size, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def pixel_shuffle_2x2(tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """2×2 pixel-shuffle：相邻 4 个 patch 拼到通道上，序列长度 /4、宽度 ×4。

    K3 用它把 ViT 网格对齐到更粗的视觉 token，再投影进 LLM。
    若高/宽为奇数，右侧/下侧补齐到偶数。
    """
    bsz, seq, dim = tokens.shape
    if seq != grid_h * grid_w:
        raise ValueError(f'视觉 token 数 {seq} 与网格 {grid_h}x{grid_w} 不符')
    pad_h = grid_h % 2
    pad_w = grid_w % 2
    x = tokens.view(bsz, grid_h, grid_w, dim)
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        grid_h, grid_w = x.shape[1], x.shape[2]
    x = x.view(bsz, grid_h // 2, 2, grid_w // 2, 2, dim)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(bsz, (grid_h // 2) * (grid_w // 2), dim * 4)


class MoonViTV2(nn.Module):
    """从图像/视频到 LLM hidden 的视觉塔 + MLP projector。"""

    def __init__(self, config):
        super().__init__()
        self.image_size = int(getattr(config, 'vision_image_size', 224))
        self.patch_size = int(getattr(config, 'vision_patch_size', 16))
        self.in_channels = int(getattr(config, 'vision_num_channels', 3))
        hidden = int(getattr(config, 'vision_hidden_size', 128))
        heads = int(getattr(config, 'vision_num_heads', 4))
        n_layers = int(getattr(config, 'vision_num_layers', 2))
        eps = float(getattr(config, 'rms_norm_eps', 1e-5))
        dropout = float(getattr(config, 'dropout', 0.0))
        llm_hidden = int(config.hidden_size)
        self.grid = self.image_size // self.patch_size
        if self.image_size % self.patch_size != 0:
            raise ValueError('vision_image_size 必须能被 vision_patch_size 整除')

        # 无 bias patch embedding：stride=patch_size 的卷积
        self.patch_embed = nn.Conv2d(
            self.in_channels, hidden, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid * self.grid, hidden))
        self.spatial_blocks = nn.ModuleList([
            MoonViTBlock(hidden, heads, mlp_ratio=4.0, eps=eps, dropout=dropout)
            for _ in range(n_layers)
        ])
        # 时间注意力：同一空间位置、不同帧之间交互（视频才用）
        self.temporal_blocks = nn.ModuleList([
            MoonViTBlock(hidden, heads, mlp_ratio=4.0, eps=eps, dropout=dropout)
            for _ in range(max(n_layers // 2, 1))
        ])
        self.norm = RMSNorm(hidden, eps=eps)
        # 2×2 shuffle 后通道变为 4h，再投影到 LLM
        self.projector = nn.Sequential(
            nn.Linear(hidden * 4, llm_hidden, bias=False),
            RMSNorm(llm_hidden, eps=eps),
            nn.Linear(llm_hidden, llm_hidden, bias=False),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _embed_frames(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: [B, C, H, W] -> [B, grid*grid, hidden]"""
        if pixels.shape[-2] != self.image_size or pixels.shape[-1] != self.image_size:
            pixels = F.interpolate(pixels, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        x = self.patch_embed(pixels)  # [B, H, gh, gw]
        bsz, hidden, grid_h, grid_w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, :x.shape[1]]
        for blk in self.spatial_blocks:
            x = blk(x)
        return x

    def _temporal_mix(self, x: torch.Tensor, n_frames: int) -> torch.Tensor:
        """x: [B*T, S, C] -> 按空间位置做时间注意力后再展平。"""
        bsz_t, seq, hidden = x.shape
        if n_frames <= 1:
            return x
        batch = bsz_t // n_frames
        # [B, T, S, C] -> [B*S, T, C]
        x = x.view(batch, n_frames, seq, hidden).permute(0, 2, 1, 3).reshape(batch * seq, n_frames, hidden)
        for blk in self.temporal_blocks:
            x = blk(x)
        x = x.view(batch, seq, n_frames, hidden).permute(0, 2, 1, 3).reshape(batch * n_frames, seq, hidden)
        return x

    def encode(self, pixel_values: Optional[torch.Tensor] = None,
               pixel_values_videos: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pixel_values_videos is not None:
            # [B, T, C, H, W]
            bsz, n_frames, channels, height, width = pixel_values_videos.shape
            frames = pixel_values_videos.reshape(bsz * n_frames, channels, height, width)
            tokens = self._embed_frames(frames)
            tokens = self._temporal_mix(tokens, n_frames)
            tokens = self.norm(tokens)
            # 每帧各自 2×2 shuffle，再沿序列拼起来
            shuffled = []
            grid = int(tokens.shape[1] ** 0.5)
            tokens = tokens.view(bsz, n_frames, tokens.shape[1], tokens.shape[2])
            for t in range(n_frames):
                shuffled.append(pixel_shuffle_2x2(tokens[:, t], grid, grid))
            tokens = torch.cat(shuffled, dim=1)
            return self.projector(tokens)

        if pixel_values is None:
            raise ValueError('MoonViT-V2 需要 pixel_values 或 pixel_values_videos')
        tokens = self._embed_frames(pixel_values)
        tokens = self.norm(tokens)
        grid = int(tokens.shape[1] ** 0.5)
        tokens = pixel_shuffle_2x2(tokens, grid, grid)
        return self.projector(tokens)
