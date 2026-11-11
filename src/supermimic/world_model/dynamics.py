"""Dynamics MLP f_omega (paper Table A.4: MLP (2048, 2048, 2048))."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.SiLU, zero_last: bool = False, last_gain: float | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsMLP(nn.Module):
    """delta = f(canonical(s), a, xi_norm) -> (…, 70). Zero-initialised last layer (identity dynamics at init)."""

    def __init__(self, feat_dim: int = 71, act_dim: int = 29, xi_dim: int = 33, hidden=(2048, 2048, 2048), out_dim: int = 70):
        super().__init__()
        self.mlp = MLP(feat_dim + act_dim + xi_dim, tuple(hidden), out_dim, zero_last=True)

    def forward(self, feat: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, action, xi_norm], dim=-1))




class HistCache(NamedTuple):
    """History of transition inputs, ordered from oldest to newest."""

    frames: torch.Tensor  # (B, Wc, feat_dim + act_dim)
    valid: torch.Tensor  # (B, Wc), bool


class HistoryMLP(nn.Module):
    """Predict a delta from stacked history, the current input and xi.

    Call init_cache once at the start of a rollout, then step_cache each step.
    The inputs are canonical features and executed actions, not raw states or
    policy observation histories. The returned delta is integrated by WorldModel.
    """

    needs_context = True

    def __init__(
        self,
        window: int = 6,
        feat_dim: int = 71,
        act_dim: int = 29,
        xi_dim: int = 33,
        hidden=(2048, 2048, 2048),
        out_dim: int = 70,
    ):
        super().__init__()
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = window
        self.Wc = window - 1
        self.feat_dim = feat_dim
        self.act_dim = act_dim
        self.fd = feat_dim + act_dim

        # Wc 个历史帧 + Wc 个有效标记 + 当前帧 + xi。
        # 默认 W=6（5 帧历史 + 1 帧当前输入）：5*100 + 5 + 100 + 33 = 638。
        self.input_dim = self.Wc * self.fd + self.Wc + self.fd + xi_dim
        self.mlp = MLP(self.input_dim, tuple(hidden), out_dim, zero_last=True)

    def init_cache(
        self,
        ctx_feat: torch.Tensor,
        ctx_action: torch.Tensor,
        xi_norm: torch.Tensor,
        ctx_valid: torch.Tensor,
    ) -> HistCache:
        """Build cache from batch-first, logged frames strictly before t.

        ctx_feat: (B,Wc,feat_dim), canonical features.
        ctx_action: (B,Wc,act_dim), actions executed at those same frames.
        xi_norm: (B,xi_dim), reserved for the shared dynamics cache interface.
        ctx_valid: (B,Wc), bool; False marks unavailable/padded history.
        """
        if ctx_feat.ndim != 3 or ctx_feat.shape[1:] != (self.Wc, self.feat_dim):
            raise ValueError("ctx_feat must have shape (B, window-1, feat_dim)")
        B = ctx_feat.shape[0]
        if ctx_action.shape != (B, self.Wc, self.act_dim):
            raise ValueError("ctx_action must have shape (B, window-1, act_dim)")
        if ctx_valid.shape != (B, self.Wc) or ctx_valid.dtype != torch.bool:
            raise ValueError("ctx_valid must be bool with shape (B, window-1)")

        frames = torch.cat([ctx_feat, ctx_action], dim=-1)
        # masked_fill 使补齐帧彻底归零，即使填充值含 NaN 也不会泄漏。
        frames = frames.masked_fill(~ctx_valid[..., None], 0.0)
        return HistCache(frames, ctx_valid)

    def step_cache(
        self,
        feat: torch.Tensor,
        action: torch.Tensor,
        xi_norm: torch.Tensor,
        cache: HistCache,
    ) -> tuple[torch.Tensor, HistCache]:
        """Return (delta, new_cache) for one autoregressive step.

        feat: (B,feat_dim); action: (B,act_dim); xi_norm: (B,xi_dim).
        The cache is updated with the current (feat_t, action_t), preserving
        gradients through predicted states and actions across rollout steps.
        """
        cur = torch.cat([feat, action], dim=-1)
        x = torch.cat(
            [cache.frames.flatten(1), cache.valid.to(cur.dtype), cur, xi_norm],
            dim=-1,
        )
        delta = self.mlp(x)

        # window=1 时历史为空，保持缓存形状不变。
        if self.Wc == 0:
            return delta, cache

        # 丢弃最旧帧，追加当前输入；不要 detach 或原地改写旧 cache。
        new_cache = HistCache(
            frames=torch.cat([cache.frames[:, 1:], cur[:, None]], dim=1),
            valid=torch.cat(
                [cache.valid[:, 1:], cache.valid.new_ones(cur.shape[0], 1)],
                dim=1,
            ),
        )
        return delta, new_cache
# R1 ADD END
