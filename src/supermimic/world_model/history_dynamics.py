"""R1: history-stacked MLP dynamics."""
from __future__ import annotations

import torch
import torch.nn as nn

from supermimic.world_model.dynamics import MLP


# R1 新增：单个 MLP 使用历史状态特征及执行动作，不包含注意力分支。
class HistoryMLP(nn.Module):
    needs_context = True

    def __init__(self, window=6, feat_dim=71, act_dim=29, xi_dim=33, hidden=(2048, 2048, 2048), out_dim=70):
        super().__init__()
        assert window >= 2
        self.window = window
        self.context_len = window - 1
        self.fd = feat_dim + act_dim
        # R1：默认 W=6（5 帧历史 + 1 帧当前），输入 638 维；输出层零初始化。
        in_dim = self.context_len * self.fd + self.context_len + self.fd + xi_dim
        self.mlp = MLP(in_dim, tuple(hidden), out_dim, zero_last=True)

    # R1：禁止绕过历史缓存调用模型。
    def forward(self, feat, action, xi_norm):
        raise RuntimeError("HistoryMLP needs a history cache: use step_cache(...)")

    # R1：tuple 中只包含固定形状张量，便于 torch.compile 追踪。
    def init_cache(self, ctx_feat, ctx_action, xi_norm, ctx_valid):
        assert ctx_feat.shape[1] == self.context_len, (ctx_feat.shape, self.context_len)
        frames = torch.cat([ctx_feat, ctx_action], dim=-1) * ctx_valid[..., None]
        return frames, ctx_valid

    # R1：预测后移除最旧帧、追加当前输入帧，缓存不 detach。
    def step_cache(self, feat, action, xi_norm, cache):
        frames, valid = cache
        cur = torch.cat([feat, action], dim=-1)
        x = torch.cat([frames.flatten(1), valid.to(cur.dtype), cur, xi_norm], dim=-1)
        new = (
            torch.cat([frames[:, 1:], cur[:, None]], dim=1),
            torch.cat([valid[:, 1:], valid.new_ones(valid.shape[0], 1)], dim=1),
        )
        return self.mlp(x), new
