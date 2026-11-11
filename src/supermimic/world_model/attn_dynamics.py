"""R2/R3/R2h/R3h: delta-space MLP + spatial/temporal attention dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from supermimic.world_model.d4_blocks import (
    BlockCausalTransformer, Modality, RMSNorm, TokenLayout,
)
from supermimic.world_model.dynamics import DynamicsMLP
# R3 add：复用 R1 历史 MLP，组成双分支模型。
from supermimic.world_model.history_dynamics import HistoryMLP
from supermimic.world_model.state_spec import (
    FEAT_DIM, FEAT_ROOT, FEAT_JPOS, FEAT_JVEL, NJ,
    XI_FRICTION, XI_ENC_BIAS, XI_COM,
)


@dataclass
class AttnDynCfg:
    """默认 R2；R3h 设置 depth=3、time_every=3、window=6。"""

    d_model: int = 128
    n_heads: int = 4
    depth: int = 3
    time_every: int = 4
    n_register: int = 0
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # R3 add：由 WorldModelCfg.history_window 统一覆盖。
    window: int = 6


def _check_inputs(feat, action, xi_norm):
    """在展平之前验证形状，避免不同样本排列被静默配对。"""
    for name, value, width in (
        ("feat", feat, FEAT_DIM),
        ("action", action, NJ),
        ("xi_norm", xi_norm, NJ + 4),
    ):
        if value.ndim < 1 or value.shape[-1] != width:
            raise ValueError(f"{name} must have shape (..., {width}); got {value.shape}")
    if action.shape[:-1] != feat.shape[:-1] or xi_norm.shape[:-1] != feat.shape[:-1]:
        raise ValueError(
            "feat, action and xi_norm must have identical leading dimensions; "
            "expand shared xi explicitly before calling the dynamics model."
        )


def joint_features(feat, action, xi_norm):
    """构造 (..., NJ, 4)：关节位置、速度、对应动作、编码器偏差。"""
    _check_inputs(feat, action, xi_norm)
    return torch.stack(
        [feat[..., FEAT_JPOS], feat[..., FEAT_JVEL], action, xi_norm[..., XI_ENC_BIAS]],
        dim=-1,
    )


def scatter_delta(root_out, joint_out):
    """根部 (...,12) 和关节 (...,NJ,2) 输出拼成现有 delta 顺序。

    顺序为 dv_h(3), dw_b(3), dqd(NJ), dp_h(3), dth_b(3), dq(NJ)。
    这个顺序与 loss 的 error_vector 不同。
    """
    return torch.cat(
        [root_out[..., :3], root_out[..., 3:6], joint_out[..., 0],
         root_out[..., 6:9], root_out[..., 9:12], joint_out[..., 1]],
        dim=-1,
    )


class TokenizedDelta(nn.Module):
    """将数值输入转换为 token，通过空间/时间注意力预测 delta 修正量。"""

    def __init__(self, cfg: AttnDynCfg):
        super().__init__()
        if cfg.d_model <= 0 or cfg.n_heads <= 0 or cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be positive and divisible by positive n_heads")
        # R3 modify：允许末层时间注意力；禁止会破坏逐帧 KV 缓存的多时间层结构。
        if cfg.depth < 1 or cfg.time_every < 1:
            raise ValueError('depth and time_every must be positive')
        if cfg.window < 2:
            raise ValueError('window must be at least 2')
        time_layers = [i for i in range(cfg.depth) if (i + 1) % cfg.time_every == 0]
        if time_layers not in ([], [cfg.depth - 1]):
            raise ValueError('temporal sublayer must occur only in the last layer')
        self.has_temporal = bool(time_layers)
        if cfg.n_register < 0 or int(cfg.d_model * cfg.mlp_ratio) < 1:
            raise ValueError("n_register must be nonnegative and MLP hidden size positive")
        if not 0 <= cfg.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.cfg = cfg
        d = cfg.d_model
        self.root_in = nn.Linear(FEAT_ROOT.stop - FEAT_ROOT.start, d)
        self.joint_in = nn.Linear(4, d)
        self.env_in = nn.Linear(4, d)

        # 每个槽位一个可学习 embedding，编码根部、各关节及环境 token 的身份。
        s = 1 + NJ + 1 + cfg.n_register
        # R3 add：并行窗口与 KV 缓存共用 token 数量。
        self.S = s
        self.slot_embed = nn.Parameter(torch.empty(s, d))
        nn.init.normal_(self.slot_embed, std=0.02)
        if cfg.n_register:
            self.registers = nn.Parameter(torch.empty(cfg.n_register, d))
            nn.init.normal_(self.registers, std=0.02)
        else:
            self.register_parameter("registers", None)

        segments = [(Modality.PROPRIO, 1), (Modality.SPATIAL, NJ), (Modality.ENV_PARAM, 1)]
        if cfg.n_register:
            segments.append((Modality.REGISTER, cfg.n_register))
        layout = TokenLayout(n_latents=0, segments=tuple(segments))
        self.transformer = BlockCausalTransformer(
            d_model=d, n_heads=cfg.n_heads, depth=cfg.depth,
            n_latents=0, modality_ids=layout.modality_ids(), space_mode="wm_agent",
            dropout=cfg.dropout, mlp_ratio=cfg.mlp_ratio,
            time_every=cfg.time_every, latents_only_time=False,
            window=cfg.window,  # R3 add：时间窗口传入 Transformer。
        )
        self.final_norm = RMSNorm(d)
        self.root_head = nn.Linear(d, 12)
        self.joint_head = nn.Linear(d, 2)
        # 初始修正量为零。输出头先获得梯度，更新后主干才能获得梯度。
        for head in (self.root_head, self.joint_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    # R3 modify：从原 forward 抽出 token 编码，单步与窗口路径共用。
    def _tokens(self, feat, action, xi_norm):
        n = feat.shape[0]

        root = self.root_in(feat[..., FEAT_ROOT])[:, None]
        joints = self.joint_in(joint_features(feat, action, xi_norm))
        env_values = torch.cat([xi_norm[..., XI_FRICTION], xi_norm[..., XI_COM]], dim=-1)
        env = self.env_in(env_values)[:, None]
        parts = [root, joints, env]
        if self.registers is not None:
            parts.append(self.registers.unsqueeze(0).expand(n, -1, -1))
        tok = torch.cat(parts, dim=1)
        tok = tok + self.slot_embed.to(dtype=tok.dtype).unsqueeze(0)

        return tok

    # R3 modify：从原 forward 抽出输出头，保留原 delta 顺序。
    def _heads(self, x):
        x = self.final_norm(x)
        return scatter_delta(self.root_head(x[:, 0]), self.joint_head(x[:, 1:1 + NJ]))

    # R3 modify：空间模型保持原入口；时间模型显式要求历史，避免静默丢失上下文。
    def forward(self, feat, action, xi_norm):
        if self.has_temporal:
            raise RuntimeError('temporal dynamics needs step_cache or forward_window')
        _check_inputs(feat, action, xi_norm)
        lead = feat.shape[:-1]
        tok = self._tokens(feat.reshape(-1, feat.shape[-1]),
                           action.reshape(-1, action.shape[-1]),
                           xi_norm.reshape(-1, xi_norm.shape[-1]))
        delta = self._heads(self.transformer(tok.unsqueeze(1)).squeeze(1))
        return delta.reshape(*lead, delta.shape[-1])

    # R3 add：并行因果参考路径，用于验证缓存数值一致性。
    def forward_window(self, feat, action, xi_norm, key_valid=None):
        # Parallel reference for the attention branch, not a full teacher-forced R3h loss.
        _check_inputs(feat, action, xi_norm)
        if feat.ndim != 3:
            raise ValueError('forward_window requires (B,T,F)')
        B, T, _ = feat.shape
        tok = self._tokens(feat.reshape(B * T, -1), action.reshape(B * T, -1),
                           xi_norm.reshape(B * T, -1))
        x = self.transformer(tok.reshape(B, T, self.S, -1), key_valid)
        return self._heads(x.reshape(B * T, self.S, -1)).reshape(B, T, -1)

    # R3 add：只编码到末层时间子层之前，缓存各帧独立的 K/V。
    def encode_context(self, ctx_feat, ctx_action, xi_norm, ctx_valid):
        if not self.has_temporal:
            raise RuntimeError('spatial-only attention has no KV cache')
        B, Wc, _ = ctx_feat.shape
        if Wc != self.cfg.window - 1:
            raise ValueError('context length must equal window - 1')
        if ctx_valid.shape != (B, Wc) or ctx_valid.dtype != torch.bool:
            raise ValueError('ctx_valid must be bool (B,W-1)')
        xi = xi_norm[:, None].expand(-1, Wc, -1)
        _check_inputs(ctx_feat, ctx_action, xi)
        tok = self._tokens(ctx_feat.reshape(B * Wc, -1), ctx_action.reshape(B * Wc, -1),
                           xi.reshape(B * Wc, -1))
        x = tok.reshape(B, Wc, self.S, -1)
        for layer in self.transformer.layers[:-1]:
            x = layer(x)
        last = self.transformer.layers[-1]
        x = last.spatial(x)
        h = last.norm2(x).permute(0, 2, 1, 3).reshape(B * self.S, Wc, -1)
        _, k, v = last.time.attn.qkv_heads(h)
        return k, v, ctx_valid

    # R3 add：固定形状空缓存；autocast 下显式传入投影 dtype。
    def empty_cache(self, B, device, dtype=None):
        if not self.has_temporal:
            raise RuntimeError('spatial-only attention has no KV cache')
        attn = self.transformer.layers[-1].time.attn
        dtype = dtype if dtype is not None else attn.qkv.weight.dtype
        z = torch.zeros(B * self.S, attn.n_heads, self.cfg.window - 1, attn.head_dim,
                        device=device, dtype=dtype)
        return z, z.clone(), torch.zeros(B, self.cfg.window - 1, dtype=torch.bool, device=device)

    # R3 add：追加当前输入帧并滚动历史；不 detach，保留跨步梯度。
    def step_cache(self, feat, action, xi_norm, cache):
        _check_inputs(feat, action, xi_norm)
        if not self.has_temporal:
            return self.forward(feat, action, xi_norm), cache
        x = self._tokens(feat, action, xi_norm)
        for layer in self.transformer.layers[:-1]:
            x = layer(x[:, None])[:, 0]
        last = self.transformer.layers[-1]
        x = last.spatial(x[:, None])[:, 0]
        y, new = last.time.step(last.norm2(x), cache)
        return self._heads(last.finish(x + last.drop2(y))), new


class HybridDynamics(DynamicsMLP):
    """继承原模型以保留 mlp.net.* 权重键；新增参数位于 attn.*。

    冻结原 MLP 时只操作 self.mlp.parameters()，避免冻结注意力分支。
    """

    def __init__(self, hidden=(2048, 2048, 2048), attn_cfg: AttnDynCfg | None = None, **kw):
        super().__init__(hidden=hidden, **kw)
        self.attn = TokenizedDelta(attn_cfg if attn_cfg is not None else AttnDynCfg())

    def forward(self, feat, action, xi_norm):
        _check_inputs(feat, action, xi_norm)
        return super().forward(feat, action, xi_norm) + self.attn(feat, action, xi_norm)

    @property
    # R3 add：向采样器声明历史长度。
    def context_len(self):
        return self.attn.cfg.window - 1 if self.attn.has_temporal else 0

    # R3 add：从 replay 历史初始化对应分支的缓存。
    def init_cache(self, ctx_feat, ctx_action, xi_norm, ctx_valid):
        return self.attn.encode_context(ctx_feat, ctx_action, xi_norm, ctx_valid)

    # R3 add：追加当前输入帧并滚动历史；不 detach，保留跨步梯度。
    def step_cache(self, feat, action, xi_norm, cache):
        d_mlp = DynamicsMLP.forward(self, feat, action, xi_norm)
        d_attn, new = self.attn.step_cache(feat, action, xi_norm, cache)
        return d_mlp + d_attn, new


# R3 add：R2h/R3h 共用的双分支封装，两个 delta 相加后由 WorldModel 积分。
class HybridHistoryDynamics(HistoryMLP):
    """R2h/R3h: HistoryMLP + spatial or spatial/temporal attention, summed before integrate."""

    def __init__(self, window=6, hidden=(2048, 2048, 2048), attn_cfg=None):
        super().__init__(window=window, hidden=hidden)
        attn_cfg = attn_cfg if attn_cfg is not None else AttnDynCfg(window=window)
        if attn_cfg.window != window:
            raise ValueError('HistoryMLP and attention windows must match')
        self.attn = TokenizedDelta(attn_cfg)

    # R3 add：从 replay 历史初始化对应分支的缓存。
    def init_cache(self, ctx_feat, ctx_action, xi_norm, ctx_valid):
        h = HistoryMLP.init_cache(self, ctx_feat, ctx_action, xi_norm, ctx_valid)
        t = self.attn.encode_context(ctx_feat, ctx_action, xi_norm, ctx_valid) if self.attn.has_temporal else None
        return h, t

    # R3 add：追加当前输入帧并滚动历史；不 detach，保留跨步梯度。
    def step_cache(self, feat, action, xi_norm, cache):
        h, t = cache
        d_mlp, h = HistoryMLP.step_cache(self, feat, action, xi_norm, h)
        d_attn, t = self.attn.step_cache(feat, action, xi_norm, t)
        return d_mlp + d_attn, (h, t)
