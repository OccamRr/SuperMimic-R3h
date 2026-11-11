"""SuperMimic R2 所需的 Dreamer 4 Transformer 模块。

输入和输出形状均为 (B, T, S, D)。R2 推荐 depth=3、time_every=4，
space_mode="wm_agent"、n_latents=0、latents_only_time=False。
本模块不自动添加位置编码；slot embedding 由 attn_dynamics.py 添加。
R3 modify：时间注意力支持相对位置偏置、历史有效掩码和固定长度 KV 缓存。

提取来源：用户提供的 dreamer4/model.py（含中文注释）。
上游 commit：未核实，使用下方源文件 SHA-256 标识本地版本。
改动：MLP 重命名为 SwiGLUMLP；增加 ENV_PARAM；全可见时省略 mask。
"""
# Source SHA-256: 7f3eb0697ca3bfd870ee5f22f83147927052ac26f93bda18bc25bf69cd89d9b8

# MIT License
# 
# Copyright (c) 2026 Nicklas Hansen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Modality', 'TokenLayout', 'RMSNorm', 'SwiGLUMLP', 'MultiheadSelfAttention', 'SpaceSelfAttentionModality', 'TimeSelfAttention', 'BlockCausalLayer', 'BlockCausalTransformer']


class Modality(IntEnum):
    LATENT = -1
    IMAGE = 0
    ACTION = 1
    PROPRIO = 2
    REGISTER = 3
    SPATIAL = 4
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP = 6
    AGENT = 7
    ENV_PARAM = 8


@dataclass(frozen=True)
class TokenLayout:
    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]

    def S(self) -> int:
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> torch.Tensor:
        parts = []
        if self.n_latents > 0:
            parts.append(torch.full((self.n_latents,), int(Modality.LATENT), dtype=torch.int32))
        for m, n in self.segments:
            if n > 0:
                parts.append(torch.full((n,), int(m), dtype=torch.int32))
        return torch.cat(parts, dim=0) if parts else torch.zeros((0,), dtype=torch.int32)

    def slices(self) -> Dict[Modality, slice]:
        idx = 0
        out: Dict[Modality, slice] = {}
        if self.n_latents > 0:
            out[Modality.LATENT] = slice(idx, idx + self.n_latents)
            idx += self.n_latents
        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n
        return out


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * (self.scale / torch.sqrt(var + self.eps))


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc_in = nn.Linear(d_model, 2 * hidden)
        self.fc_out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc_in(x).chunk(2, dim=-1)
        h = u * F.silu(v)
        h = self.drop(h)
        y = self.fc_out(h)
        y = self.drop(y)
        return y


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    # R3 add
    def qkv_heads(self, x):
        # (N,L,D) -> 三个 (N,H,L,head_dim)
        N, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def heads(t):
            return t.view(
                N, L, self.n_heads, self.head_dim
            ).transpose(1, 2)

        return heads(q), heads(k), heads(v)


    # R3 add：分离注意力计算，单步 query 可以读取缓存的 K/V。
    def attend(self, q, k, v, *, attn_mask=None, is_causal=False):
        N, H, L, hd = q.shape
        drop = self.dropout_p if self.training else 0.0

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=drop,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(N, L, H * hd)
        return self.out(y)

    # R3 modify
    def forward(self, x_nld, *, attn_mask=None, is_causal=False):
        q, k, v = self.qkv_heads(x_nld)
        return self.attend(
            q, k, v,
            attn_mask=attn_mask,
               is_causal=is_causal,
        )



class SpaceSelfAttentionModality(nn.Module):
    def __init__(self, d_model: int, n_heads: int, modality_ids: torch.Tensor, n_latents: int, mode: str, dropout: float):
        super().__init__()
        self.n_latents = int(n_latents)
        self.mode = mode
        self.register_buffer("modality_ids", modality_ids.to(torch.int32), persistent=False)

        S = int(self.modality_ids.numel())
        allow = self._build_allow(S)                               # (S,S) True=allowed
        self.full_attention = bool(allow.all().item())
        attn_mask = allow.unsqueeze(0).unsqueeze(0)                # (1,1,S,S) True=allowed (PyTorch SDPA bool mask)
        self.register_buffer("attn_mask", attn_mask, persistent=False)

        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def _build_allow(self, S: int) -> torch.Tensor:
        device = self.modality_ids.device
        q_idx = torch.arange(S, device=device).unsqueeze(1)  # (S,1)
        k_idx = torch.arange(S, device=device).unsqueeze(0)  # (1,S)

        is_q_lat = q_idx < self.n_latents
        is_k_lat = k_idx < self.n_latents

        q_mod = self.modality_ids[q_idx]
        k_mod = self.modality_ids[k_idx]
        same_mod = (q_mod == k_mod)

        if self.mode == "encoder":
            allow_lat_q = torch.ones((S, S), dtype=torch.bool, device=device)
            allow_nonlat_q = same_mod
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        if self.mode == "decoder":
            allow_lat_q = is_k_lat
            allow_nonlat_q = same_mod | is_k_lat
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        if self.mode == "wm_agent":
            # full mixing across modalities
            return torch.ones((S, S), dtype=torch.bool, device=device)

        if self.mode == "wm_agent_isolated":
            # non-agent tokens: can attend to everything EXCEPT agent tokens
            # agent tokens: attend only to agent tokens (keeps them inert in pretrain)
            is_q_agent = (q_mod == int(Modality.AGENT))
            is_k_agent = (k_mod == int(Modality.AGENT))

            allow = torch.ones((S, S), dtype=torch.bool, device=device)

            # non-agent queries cannot see agent keys
            allow_non_agent_q = ~is_q_agent
            allow = torch.where(allow_non_agent_q, ~is_k_agent, allow)

            # agent queries only see agent keys
            allow = torch.where(is_q_agent, is_k_agent, allow)
            return allow

        raise ValueError(f"Unsupported mode for tokenizer/wm: {self.mode}")

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        x = x_btSd.reshape(B * T, S, D)
        mask = None if self.full_attention else self.attn_mask.expand(B * T, 1, S, S)
        y = self.attn(x, attn_mask=mask, is_causal=False)
        return y.reshape(B, T, S, D)


# R3 modify：支持带相对位置偏置的并行因果计算与单步 KV 缓存。
class TimeSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout,
        latents_only,
        n_latents,
        window=0,
    ):
        super().__init__()
        self.latents_only = bool(latents_only)
        self.n_latents = int(n_latents)
        self.window = window

        self.attn = MultiheadSelfAttention(
            d_model, n_heads, dropout=dropout
        )

        self.rel_bias = (
            nn.Parameter(torch.zeros(n_heads, window))
            if window > 0 else None
        )

    def forward(self, x_btSd, key_valid=None):
        B, T, S, D = x_btSd.shape

        source = (
            x_btSd[:, :, :self.n_latents]
            if self.latents_only else x_btSd
        )
        count = source.shape[2]

        x = (
            source.permute(0, 2, 1, 3)
            .contiguous()
            .view(B * count, T, D)
        )

        if self.rel_bias is None and key_valid is None:
            y = self.attn(x, is_causal=True)
        else:
            q, k, v = self.attn.qkv_heads(x)
            mask = self._parallel_mask(
                B, count, T, key_valid, q
            )
            y = self.attn.attend(
                q, k, v, attn_mask=mask
            )

        y = (
            y.view(B, count, T, D)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        if self.latents_only:
            result = x_btSd.clone()
            result[:, :, :self.n_latents] = y
            return result

        return y

    # R3 add：无效 key 不可见，query 自身始终可见，避免全屏蔽行。
    def _parallel_mask(self, B, S, T, key_valid, ref):
        if self.rel_bias is not None and T > self.window:
            raise ValueError("T must not exceed window")

        if key_valid is not None:
            if (
                key_valid.shape != (B, T)
                or key_valid.dtype != torch.bool
            ):
                raise ValueError(
                    "key_valid must be bool (B,T)"
                )

        i = torch.arange(T, device=ref.device)[:, None]
        j = torch.arange(T, device=ref.device)[None, :]

        # 因果：只能读取当前帧和过去帧。
        allow = (j <= i)[None]

        if key_valid is not None:
            # 无效帧不能作为其他帧的 key。
            # 每个 query 至少允许读取自身，避免全屏蔽行。
            allow = allow & (
                key_valid[:, None, :]
                | (i == j)[None]
            )

        if self.rel_bias is None:
            bias = ref.new_zeros(
                self.attn.n_heads, T, T
            )
        else:
            distance = (i - j).clamp(min=0)
            bias = self.rel_bias[:, distance].to(ref.dtype)

        mask = torch.where(
            allow[:, None],
            bias[None],
            bias.new_full((), float("-inf")),
        )

        # 展平顺序对应 B*S。
        return (
            mask.expand(B, -1, -1, -1)
            .repeat_interleave(S, dim=0)
        )

    # R3 add：缓存从旧到新排列，距离每步重新计算，缓存张量不 detach。
    def step(self, h, cache):
        # h: (B,S,D)
        # ck/cv: (B*S,H,W-1,head_dim)
        # cvalid: (B,W-1)
        if self.latents_only or self.rel_bias is None:
            raise ValueError(
                "cached attention requires all tokens "
                "and a positive window"
            )

        ck, cv, cvalid = cache
        B, S, D = h.shape

        q, k, v = self.attn.qkv_heads(
            h.reshape(B * S, 1, D)
        )

        k_all = torch.cat([ck, k], dim=2)
        v_all = torch.cat([cv, v], dim=2)

        valid = torch.cat(
            [cvalid, cvalid.new_ones(B, 1)],
            dim=1,
        )

        # key 从最旧到最新：
        # 相对当前 query 的距离为 W-1,...,1,0。
        distance = torch.arange(
            self.window - 1, -1, -1,
            device=h.device,
        )

        bias = self.rel_bias[:, distance].to(q.dtype)
        bias = bias[None, :, None, :]

        mask = torch.where(
            valid.repeat_interleave(S, dim=0)[
                :, None, None, :
            ],
            bias,
            bias.new_full((), float("-inf")),
        )

        y = self.attn.attend(
            q, k_all, v_all, attn_mask=mask
        )

        # 丢弃最旧帧；保留当前帧供下一步使用。
        # 不 detach，保持跨步梯度。
        new_cache = (
            k_all[:, :, 1:],
            v_all[:, :, 1:],
            valid[:, 1:],
        )

        return y.reshape(B, S, D), new_cache


class BlockCausalLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        layer_index: int,
        time_every: int,
        latents_only_time: bool,
        
        window: int = 0, # R3 add
    ):
        super().__init__()
        self.do_time = ((layer_index + 1) % time_every == 0)

        self.norm1 = RMSNorm(d_model)
        self.space = SpaceSelfAttentionModality(d_model, n_heads, modality_ids, n_latents, space_mode, dropout)
        self.drop1 = nn.Dropout(dropout)

        if self.do_time:
            self.norm2 = RMSNorm(d_model)
            # R3 modify：仅时间层创建模块，并传入窗口，避免空间层出现闲置参数。
            self.time = TimeSelfAttention(
                d_model, n_heads, dropout, latents_only_time, n_latents, window=window,
            )
            self.drop2 = nn.Dropout(dropout)

        self.norm3 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    # R3 add：拆出空间子层，历史编码停在时间子层之前。
    def spatial(self, x):
        return x + self.drop1(
            self.space(self.norm1(x))
        )


    # R3 add：共用末尾 MLP 子层，避免缓存与并行路径各自实现。
    def finish(self, x):
        return x + self.mlp(self.norm3(x))


    # R3 modify：逐层传递历史有效标记。
    def forward(self, x, key_valid=None):
        x = self.spatial(x)

        if self.do_time:
            x = x + self.drop2(
                self.time(self.norm2(x), key_valid)
            )

        return self.finish(x)


class BlockCausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        time_every: int,
        latents_only_time: bool,
        # R3 add
        window: int = 0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BlockCausalLayer(
                d_model=d_model, n_heads=n_heads, n_latents=n_latents,
                modality_ids=modality_ids, space_mode=space_mode,
                dropout=dropout, mlp_ratio=mlp_ratio,
                layer_index=i, time_every=time_every,
                latents_only_time=latents_only_time,
                # R3 add
                window=window,
            )
            for i in range(depth)
        ])
        
    # R3 modify：逐层传递历史有效标记。
    def forward(self, x, key_valid=None):
        for layer in self.layers:
            x = layer(x, key_valid)
        return x