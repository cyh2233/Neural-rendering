"""Visual comparisons: held-out grids, object crops and multi-panel trajectory videos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nr.data.cache import load_image
from nr.data.types import Camera
from nr.eval.nvs import object_boxes, save_frames
from nr.utils.geometry import lateral_shift


def to_u8(x: torch.Tensor) -> np.ndarray:
    return (x.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


def enlarge(img: np.ndarray, height: int) -> np.ndarray:
    k = max(1, height // img.shape[0])
    return img.repeat(k, 0).repeat(k, 1)


def _pad_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.full((h, w, 3), 255, np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    return out


def _stack_rows(rows: list[np.ndarray], gap: int = 4) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    sep = np.full((gap, w, 3), 255, np.uint8)
    parts = []
    for r in rows:
        parts += [_pad_to(r, r.shape[0], w), sep]
    return np.concatenate(parts[:-1], 0)


@torch.no_grad()
def write_comparisons(graph, views, backend: str, out_prefix: str | Path, lateral: float = -1.5,
                      max_views: int = 6, crop_height: int = 160) -> list[Path]:
    """Writes ``<prefix>_frames.png`` (ground truth | render | background only | shifted view) and,
    if any dynamic object is visible, ``<prefix>_crops.png`` (ground-truth crop | render crop)."""
    import imageio.v3 as iio

    with_obj = [(c, im) for c, im in views if object_boxes(graph, c)]
    chosen = with_obj[:max_views] or views[:max_views]
    rows, crops = [], []
    for cam, img in chosen:
        gt = img.numpy() if img.dtype == torch.uint8 else to_u8(img)
        full = to_u8(graph.render(cam, backend=backend)["rgb"])
        bg = to_u8(graph.render(cam, backend=backend, include_objects=False)["rgb"])
        shifted = to_u8(graph.render(cam.with_pose(lateral_shift(cam.c2w, lateral)), backend=backend)["rgb"])
        rows.append(np.concatenate([gt, full, bg, shifted], 1))
        for x0, y0, x1, y1 in object_boxes(graph, cam):
            gap = np.full((y1 - y0, 2, 3), 255, np.uint8)
            crops.append(enlarge(np.concatenate([gt[y0:y1, x0:x1], gap, full[y0:y1, x0:x1]], 1), crop_height))
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    written = []
    if rows:
        frames = _stack_rows(rows)
        if frames.shape[0] < 400:
            frames = enlarge(frames, 2 * frames.shape[0])
        path = Path(f"{out_prefix}_frames.png")
        iio.imwrite(path, frames)
        written.append(path)
    if crops:
        path = Path(f"{out_prefix}_crops.png")
        iio.imwrite(path, _stack_rows(crops, gap=6))
        written.append(path)
    return written


@torch.no_grad()
def write_trajectory_video(graph, cams: list[Camera], backend: str, out: str | Path, image_scale: float = 1.0,
                           lateral: float = 1.5, fps: int = 4) -> Path:
    """2x2 video over one camera's trajectory: [ground truth | reconstruction] on top,
    [shifted left | shifted right] by ``lateral`` metres below."""
    cams = sorted(cams, key=lambda c: c.frame_idx)
    frames = []
    for cam in cams:
        gt = to_u8(load_image(cam, image_scale))
        c = cam.scaled(image_scale)
        rec = to_u8(graph.render(c, backend=backend)["rgb"])
        left = to_u8(graph.render(c.with_pose(lateral_shift(c.c2w, -lateral)), backend=backend)["rgb"])
        right = to_u8(graph.render(c.with_pose(lateral_shift(c.c2w, lateral)), backend=backend)["rgb"])
        h, w = rec.shape[:2]
        gt = _pad_to(gt[:h, :w], h, w)
        frames.append(np.concatenate([np.concatenate([gt, rec], 1), np.concatenate([left, right], 1)], 0))
    return save_frames(frames, out, fps=fps)
