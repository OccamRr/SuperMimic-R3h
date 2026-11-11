"""World model: learned dynamics + learned termination head + analytic observation/reward heads."""

from __future__ import annotations

# R3 modify：replace 将窗口配置统一传入注意力分支。
from dataclasses import dataclass, field, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from supermimic.world_model.dynamics import MLP, DynamicsMLP
# R1：独立历史动力学模型，按配置选择。
from supermimic.world_model.history_dynamics import HistoryMLP
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


# R3 modify：增加历史 MLP + 注意力双分支模型。
from supermimic.world_model.attn_dynamics import AttnDynCfg, HybridDynamics, HybridHistoryDynamics

@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    term_hidden: tuple[int, ...] = (64, 64)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8
    dynamics_type: str = "mlp"
    attn: AttnDynCfg = field(default_factory=AttnDynCfg)
    # R1：默认 5 帧历史 + 1 帧当前；历史缓存长度为 W - 1。
    history_window: int = 6



class TerminationHead(nn.Module):
    """P(terminal | state, reference). Input: analytic termination features and their magnitudes."""

    def __init__(self, hidden=(64, 64), height_thr: float = 0.25, tilt_thr: float = 0.8):
        super().__init__()
        self.mlp = MLP(4, tuple(hidden), 1)
        self.register_buffer("scale", torch.tensor([1.0 / height_thr, 1.0 / tilt_thr]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (…, 2) = [height error, tilt error] -> logits (…,)"""
        x = feats * self.scale
        return self.mlp(torch.cat([x, x.abs()], dim=-1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device):
        super().__init__()
        self.cfg = cfg
        self.spec = StateSpec(cfg.state_spec, device)

        # R3 add：history_window 是 MLP、注意力和采样缓存的唯一窗口来源。
        attn_cfg = replace(cfg.attn, window=cfg.history_window)
        if cfg.dynamics_type == "mlp":
            self.dynamics = DynamicsMLP(hidden=cfg.hidden)
        elif cfg.dynamics_type == "hybrid":
            self.dynamics = HybridDynamics(
                 hidden=cfg.hidden,
                 attn_cfg=attn_cfg,  # R3 modify：使用统一窗口配置。
            )
        # R1：此分支仅创建 HistoryMLP，不包含空间注意力。
        elif cfg.dynamics_type == "history_mlp":
            self.dynamics = HistoryMLP(window=cfg.history_window, hidden=cfg.hidden)
        # R3 add：time_every=3 为 R3h，time_every>depth 为 R2h。
        elif cfg.dynamics_type == "hybrid_history":
            self.dynamics = HybridHistoryDynamics(
                window=cfg.history_window, hidden=cfg.hidden, attn_cfg=attn_cfg,
            )
        else:
            raise ValueError(f"unknown dynamics_type: {cfg.dynamics_type}")

        self.term_head = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
        self.heads = heads  # analytic observation / reward / FK (no parameters)
        self.to(device)

    # R1：历史长度以模型为唯一来源，供训练及策略采样共用。
    @property
    def context_len(self) -> int:
        return int(getattr(self.dynamics, "context_len", 0))

    # R1：replay 为时间优先；动力学缓存为 batch 优先。
    def init_cache(self, ctx_states, ctx_actions, xi, ctx_valid):
        cf = self.spec.canonical(ctx_states).transpose(0, 1)
        return self.dynamics.init_cache(
            cf, ctx_actions.transpose(0, 1), self.spec.normalize_xi(xi), ctx_valid.transpose(0, 1)
        )

    # R1：缓存也参与自动求导，保留跨步状态和动作的梯度。
    def step_cache(self, s, action, xi, cache):
        delta, cache = self.dynamics.step_cache(
            self.spec.canonical(s), action, self.spec.normalize_xi(xi), cache
        )
        return self.spec.integrate(s, delta), cache

    # R1：统一检查训练与评估的上下文，防止静默遗漏历史。
    def _init_ctx(self, ctx, xi):
        wc = self.context_len
        if wc == 0:
            assert ctx is None, "this dynamics model takes no context"
            return None
        assert ctx is not None and all(x is not None for x in ctx), f"dynamics needs {wc} context frames"
        cs, ca, cv = ctx
        assert cs.shape[0] == ca.shape[0] == cv.shape[0] == wc, (cs.shape, ca.shape, cv.shape, wc)
        return self.init_cache(cs, ca, xi, cv)

    # ------------------------------------------------------------------ one step

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi))."""
        feat = self.spec.canonical(s)
        delta = self.dynamics(feat, action, self.spec.normalize_xi(xi))
        return self.spec.integrate(s, delta)

    def termination_logits(self, s_next: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        # The pelvis is the free-joint body, so its pose is the state's root pose: no FK needed.
        feats = self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ref)
        return self.term_head(feats)

    # ------------------------------------------------------------------ Eq. 1

    # R1：新增 cache 参数，并将更新后的 cache 作为第五项返回。
    def _loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat, cache=None):
        """One autoregressive step of Eq. 1: returns (s_next, sq_err (n,70), term_bce (n,), logits (n,), cache)."""
        if cache is None:
            s_next = self.step(s, a, xi)
        else:
            s_next, cache = self.step_cache(s, a, xi, cache)
        e = self.spec.error_vector(s_next, s_true)
        feats = self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ReferenceFrame(None, None, None, None, ref_root_pos, ref_root_quat))
        logits = self.term_head(feats)
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none")
        return s_next, e * e, bce, logits, cache

    def enable_compile(self, mode: str | None = "default") -> None:
        self._loss_step_fn = torch.compile(self._loss_step, mode=None if mode == "default" else mode, dynamic=False) if mode else self._loss_step

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, done: torch.Tensor, refs: list[ReferenceFrame], ctx=None):
        """k-step autoregressive loss.

        states: (k+1, n, 71) logged states; a_exec: (k, n, 29); xi: (n, 33);
        done: (k, n) termination label of states 1..k; refs[j]: reference of transition j (for the
        termination features of state j+1).
        """
        k = a_exec.shape[0]
        s = states[0]
        # R1：每个窗口初始化一次历史，然后在预测循环中持续更新。
        cache = self._init_ctx(ctx, xi)
        total_state = 0.0
        total_term = 0.0
        metrics = {}
        err_first = None
        err_last = None
        acc_num = 0.0
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        for j in range(k):
            s, sq, bce, logits, cache = step_fn(s, a_exec[j], xi, states[j + 1], done[j], refs[j].root_pos_w, refs[j].root_quat_w, cache)
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + bce.mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                if j == 0:
                    err_first = nse
                err_last = nse
                acc_num += ((logits > 0).float() == done[j]).float().mean()
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        # Metrics stay on-device (detached tensors): a float() here would stall the CPU on the
        # forward pass every minibatch, which in data-parallel runs turns into lockstep waiting.
        metrics.update(
            state_loss=(total_state / k).detach(),
            term_bce=(total_term / k).detach(),
            nse_1step=err_first.detach(),
            nse_kstep=err_last.detach(),
            term_acc=(acc_num / k).detach(),
        )
        return loss, metrics

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, ctx=None) -> torch.Tensor:
        """Open-loop k-step normalised squared error per step (k,), averaged over the batch."""
        k = a_exec.shape[0]
        s = states[0]
        # R1：每个窗口初始化一次历史，然后在预测循环中持续更新。
        cache = self._init_ctx(ctx, xi)
        out = []
        for j in range(k):
            if cache is None:
                s = self.step(s, a_exec[j], xi)
            else:
                s, cache = self.step_cache(s, a_exec[j], xi, cache)
            out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)
