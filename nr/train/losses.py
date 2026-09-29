"""Training losses."""

from __future__ import annotations

import torch

from nr.eval.metrics import ssim


def photometric_loss(pred: torch.Tensor, gt: torch.Tensor, ssim_lambda: float = 0.2) -> torch.Tensor:
    """(1 - lambda) * L1 + lambda * (1 - SSIM), as in 3DGS."""
    l1 = (pred - gt).abs().mean()
    if ssim_lambda <= 0:
        return l1
    return (1 - ssim_lambda) * l1 + ssim_lambda * (1 - ssim(pred, gt))


def lidar_depth_loss(
    depth: torch.Tensor,        # (H, W, 1) rendered expected depth
    alpha: torch.Tensor,        # (H, W, 1)
    lidar_uvz: torch.Tensor,    # (M, 3) pixel u, v, metric depth
    max_depth: float = 80.0,
) -> torch.Tensor:
    """Relative L1 between rendered depth and projected LiDAR, where the render is opaque."""
    if lidar_uvz is None or lidar_uvz.numel() == 0:
        return depth.new_zeros(())
    h, w = depth.shape[:2]
    lidar_uvz = lidar_uvz.to(depth.device)
    u = lidar_uvz[:, 0].long().clamp(0, w - 1)
    v = lidar_uvz[:, 1].long().clamp(0, h - 1)
    gt = lidar_uvz[:, 2]
    pred = depth[v, u, 0]
    mask = (gt < max_depth) & (alpha[v, u, 0].detach() > 0.5)
    if not mask.any():
        return depth.new_zeros(())
    return ((pred[mask] - gt[mask]).abs() / gt[mask]).mean()


def opacity_regularizer(opacities: torch.Tensor) -> torch.Tensor:
    return opacities.mean()
