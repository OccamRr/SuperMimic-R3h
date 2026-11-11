"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.world_model.heads import ReferenceFrame
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.world_model import WorldModel


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = "default"


def reference_at(lib: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w)


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg):
        self.wm = wm
        self.lib = lib
        self.cfg = cfg
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def loss_on(self, w: WindowBatch):
        k, n = w.frame.shape
        # Only the root pose of the references is needed for the termination features.
        r = self.lib.ref(w.frame.reshape(-1))
        refs = [ReferenceFrame(None, None, None, None, r.root_pos_w[j * n : (j + 1) * n], r.root_quat_w[j * n : (j + 1) * n]) for j in range(k)]
        # R1：将采样上下文送入 k 步自回归损失。
        ctx = (w.ctx_states, w.ctx_a_exec, w.ctx_valid) if w.ctx_states is not None else None
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs, ctx=ctx)

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        for _ in range(self.cfg.batches_per_update):
            # R1：历史长度由模型声明，避免采样与模型配置不一致。
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window, ctx=self.wm.context_len)
            loss, m = self.loss_on(w)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            allreduce_grads(self.wm)
            gn = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip)
            self.opt.step()
            m["loss"] = loss.detach()
            m["grad_norm"] = gn.detach()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        # Single host sync per update (keeps the CPU queuing GPU work across minibatches).
        return {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}

    # R1：real / shuffle / zero 只干预初始上下文；后续缓存照常更新。
    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8, ctx_mode: str = "real") -> torch.Tensor:
        """k-step open-loop error; ctx_mode is real, shuffle or zero."""
        if ctx_mode not in ("real", "shuffle", "zero"):
            raise ValueError(f"unknown ctx_mode: {ctx_mode}")
        w = buffer.sample_windows(n, k, ctx=self.wm.context_len)
        ctx = None
        if w.ctx_states is not None:
            cs, ca, cv = w.ctx_states, w.ctx_a_exec, w.ctx_valid
            if ctx_mode == "shuffle":
                p = torch.randperm(n, device=cs.device)
                cs, ca, cv = cs[:, p], ca[:, p], cv[:, p]
            elif ctx_mode == "zero":
                cv = torch.zeros_like(cv)
            ctx = (cs, ca, cv)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi, ctx=ctx)
