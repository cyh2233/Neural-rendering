"""Backend-agnostic rasterization entry point."""

from __future__ import annotations

import torch

from nr.render.gsplat_backend import gsplat_available, rasterize_gsplat
from nr.render.torch_backend import rasterize_torch


def resolve_backend(name: str = "auto") -> str:
    if name == "auto":
        return "gsplat" if gsplat_available() else "torch"
    if name == "gsplat" and not gsplat_available():
        raise RuntimeError("backend 'gsplat' requested but CUDA or the gsplat package is unavailable")
    if name not in ("gsplat", "torch"):
        raise ValueError(f"unknown backend {name!r}")
    return name


def rasterize(
    means: torch.Tensor,        # (N, 3) world
    quats: torch.Tensor,        # (N, 4) wxyz, need not be normalised
    scales: torch.Tensor,       # (N, 3) activated (positive)
    opacities: torch.Tensor,    # (N,) activated, in (0, 1)
    colors: torch.Tensor,       # (N, C) view-dependent colour already evaluated
    viewmat: torch.Tensor,      # (4, 4) world-to-camera
    K: torch.Tensor,            # (3, 3)
    width: int,
    height: int,
    backend: str = "auto",
    near: float = 0.01,
    far: float = 1e10,
):
    """Returns (rgb (H,W,C) premultiplied, alpha (H,W,1), depth (H,W,1) expected depth, info)."""
    backend = resolve_backend(backend)
    fn = rasterize_gsplat if backend == "gsplat" else rasterize_torch
    return fn(means, quats, scales, opacities, colors, viewmat, K, width, height, near, far)
