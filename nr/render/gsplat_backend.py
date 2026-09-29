"""Thin adapter around ``gsplat.rasterization`` (CUDA). Same outputs as the torch backend."""

from __future__ import annotations

import torch


def gsplat_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import gsplat  # noqa: F401
    except ImportError:
        return False
    return True


def rasterize_gsplat(means, quats, scales, opacities, colors, viewmat, K, width, height, near=0.01, far=1e10):
    from gsplat import rasterization

    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        render_mode="RGB+ED",
        packed=False,
    )
    n_ch = colors.shape[-1]
    rgb = renders[0, ..., :n_ch]
    depth = renders[0, ..., n_ch : n_ch + 1]
    alpha = alphas[0]
    means2d = info["means2d"]
    if means2d.requires_grad:
        means2d.retain_grad()
    radii = info["radii"]
    if radii.dim() == 3:  # gsplat >= 1.5 returns per-axis radii (C, N, 2)
        radii = radii.max(dim=-1).values
    out = {
        "means2d": means2d,  # (1, N, 2); gradient is read by the densification strategy
        "radii": radii[0],
        "depths": info.get("depths", None),
        "width": width,
        "height": height,
    }
    return rgb, alpha, depth, out
