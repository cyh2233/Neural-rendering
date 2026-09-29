"""Held-out evaluation and novel-trajectory rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nr.data.types import Camera
from nr.eval.metrics import LPIPS, psnr, ssim
from nr.utils.geometry import lateral_shift


@torch.no_grad()
def evaluate_views(graph, views, backend: str, device, sh_degree: int | None = None, use_lpips: bool = True) -> dict:
    lp = LPIPS() if use_lpips and LPIPS.available() else None
    acc: dict[str, list[float]] = {"psnr": [], "ssim": []}
    if lp is not None:
        acc["lpips"] = []
    for cam, img in views:
        gt = img.to(device).float() / 255.0 if img.dtype == torch.uint8 else img.to(device)
        pred = graph.render(cam, sh_degree, backend)["rgb"].clamp(0, 1)
        acc["psnr"].append(float(psnr(pred, gt)))
        acc["ssim"].append(float(ssim(pred, gt)))
        if lp is not None:
            acc["lpips"].append(float(lp(pred, gt)))
    out = {k: float(np.mean(v)) for k, v in acc.items() if v}
    out["num_views"] = len(views)
    return out


@torch.no_grad()
def render_views(graph, cams: list[Camera], backend: str, lateral: float = 0.0, up: float = 0.0,
                 sh_degree: int | None = None) -> list[np.ndarray]:
    """Render each camera, optionally displaced sideways / vertically (e.g. a lane change)."""
    frames = []
    for cam in cams:
        if lateral or up:
            cam = cam.with_pose(lateral_shift(cam.c2w, lateral, up))
        rgb = graph.render(cam, sh_degree, backend)["rgb"].clamp(0, 1)
        frames.append((rgb.cpu().numpy() * 255).round().astype(np.uint8))
    return frames


def save_frames(frames: list[np.ndarray], out: str | Path, fps: int = 10) -> Path:
    """Write an .mp4 if ffmpeg is available, otherwise a directory of PNGs."""
    import imageio.v3 as iio

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix in (".mp4", ".gif"):
        try:
            if out.suffix == ".mp4":
                iio.imwrite(out, np.stack(frames), fps=fps, codec="libx264")
            else:
                iio.imwrite(out, np.stack(frames), duration=1000 / fps, loop=0)
            return out
        except Exception as e:  # missing ffmpeg plugin etc.
            print(f"could not write {out.name} ({e}); writing PNG frames instead")
            out = out.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        iio.imwrite(out / f"{i:05d}.png", f)
    return out
