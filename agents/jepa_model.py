# agents/jepa_model.py
# [JEPA] New file: lightweight I-JEPA-style model using existing ConvEncoder
#         with EMA target encoder (teacher).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# [JEPA] Reuse your existing CNN encoder; DO NOT rewrite CNN.
from .nn_actor_critic import ConvEncoder


@dataclass
class JEPAConfig:
    """Config for JEPA pretraining."""
    in_channels: int = 12          # 4-frame RGB stack => 12 channels
    feat_dim: int = 256            # must match ConvEncoder(feat_dim)
    mask_ratio: float = 0.6        # fraction of pixels roughly masked
    num_blocks: int = 4            # number of rectangular blocks per image
    var_weight: float = 1.0        # weight of variance regularization term
    covar_weight: float = 0.5      # weight of covariance regularization term
    momentum: float = 0.99         # [JEPA] EMA momentum for target encoder
    std_target: float = 1.0


class JEPAModel(nn.Module):
    """
    Simplified I-JEPA-style model built on top of your ConvEncoder:

      - Online encoder: self.encoder(x_masked)
      - Target encoder: self.target_encoder(x_full), updated via EMA
      - Predictor: MLP(feat_dim -> feat_dim)
      - Loss: MSE(predict(online_repr), stopgrad[target_repr]) + var_reg(target)

    Doing JEPA on global feature space *ConvEncoder's output)
    Totally intact on internal structure of ConvEncoder, same as the compatibility as PPO/ActorCritic.
    """

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg

        # [JEPA] Online encoder: same ConvEncoder as PPO uses.
        self.encoder = ConvEncoder(
            in_channels=cfg.in_channels,
            feat_dim=cfg.feat_dim,
        )

        # [JEPA] EMA target encoder (teacher): same architecture, no gradients.
        self.target_encoder = ConvEncoder(
            in_channels=cfg.in_channels,
            feat_dim=cfg.feat_dim,
        )
        self._init_target_encoder()  # [JEPA] EMA: copy weights and freeze

        # [JEPA] Predictor head on top of context representation.
        self.predictor = nn.Sequential(
            nn.Linear(cfg.feat_dim, cfg.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.feat_dim, cfg.feat_dim),
        )

    # ------------------------------------------------------------------
    # [JEPA] EMA helper: initialize and update target encoder
    # ------------------------------------------------------------------
    def _init_target_encoder(self) -> None:
        """[JEPA] EMA: initialize target_encoder from encoder and freeze it."""
        # Copy weights
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        # Freeze params: target_encoder 不參與 gradient，只用 EMA 更新
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self, momentum: float | None = None) -> None:
        """
        [JEPA] EMA: update target encoder parameters.

        θ_target ← m * θ_target + (1 - m) * θ_online
        """
        m = self.cfg.momentum if momentum is None else momentum
        m = float(m)
        for p_t, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            # p_t = m * p_t + (1 - m) * p
            p_t.data.mul_(m).add_(p.data, alpha=1.0 - m)

    # ------------------------------------------------------------------
    # Mask generator on input image
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _make_mask(self, batch_size: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Fast patch-wise mask:
        1. build low resolution patch mask (B, 1, H_p, W_p)
        2. use repeat_interleave to enlarge it to (B, 1, H, W)
        """
        patch_h = 4
        patch_w = 4

        H_p = h // patch_h
        W_p = w // patch_w

        # (B, 1, H_p, W_p)，use mask_ratio to mask out
        patch_mask = (torch.rand(batch_size, 1, H_p, W_p, device=device) < self.cfg.mask_ratio).float()

        # resize back to original resolution
        mask = patch_mask.repeat_interleave(patch_h, dim=2).repeat_interleave(patch_w, dim=3)

        # crop to original h, w size
        mask = mask[:, :, :h, :w]

        return mask

    # ------------------------------------------------------------------
    # Forward: compute JEPA loss
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        x_target: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute JEPA loss on a batch of images.

        Args:
            x:         (B, C, H, W), context frame at time t.
            x_target:  (B, C, H, W), target frame at time t or t+Δ.
                       If None, we use x itself as the target.
        Returns:
            total_loss: scalar
            stats: logging dict with loss terms.
        """
        # handling dtype first
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        if x_target is None:
            x_target = x
        elif x_target.dtype == torch.uint8:
            x_target = x_target.float() / 255.0

        B, C, H, W = x.shape
        device = x.device

        # -----------------------
        # Context branch: x_masked (t)
        # -----------------------
        mask = self._make_mask(B, H, W, device=device)
        x_masked = x * (1.0 - mask)
        context = self.encoder(x_masked)   # (B, feat_dim)
        pred = self.predictor(context)     # (B, feat_dim)

        # -----------------------
        # Target branch: x_target (t or t+Δ)
        # -----------------------
        with torch.no_grad():
            target = self.target_encoder(x_target)  # (B, feat_dim)
            target = target.detach()

        # ---------------------------------
        # 1) Prediction loss (MSE)
        # ---------------------------------
        recon_loss = F.mse_loss(pred, target)

        # ---------------------------------
        # 2) Variance regularization (avoid collapse)
        # ---------------------------------
        z = pred                                                # (B, feat_dim)
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0) + 1e-4)                   # (feat_dim,)
        var_loss = torch.mean(F.relu(self.cfg.std_target - std))

        # ---------------------------------
        # 3) Covariance regularization (avoid highly correlated representation)
        # ---------------------------------
        cov = (z.T @ z) / max(B - 1, 1)                         # (feat_dim, feat_dim)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag ** 2).mean()

        total_loss = (
            recon_loss
            + self.cfg.var_weight * var_loss
            + self.cfg.covar_weight * cov_loss
        )

        stats = {
            "loss_total": total_loss.detach(),
            "loss_recon": recon_loss.detach(),
            "loss_var": var_loss.detach(),
            "loss_cov": cov_loss.detach(),
            "cov_offdiag_abs_mean": off_diag.abs().mean().detach(),
            "var_abs_mean": std.abs().mean().detach(),
            "mask_ratio_effective": mask.mean().detach(),
        }
        return total_loss, stats
