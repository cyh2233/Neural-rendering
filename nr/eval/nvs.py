"""Held-out evaluation and novel-trajectory rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nr.data.types import Camera
from nr.eval.metrics import LPIPS, psnr, ssim
from nr.utils.geometry import box_corners, lateral_shift, make_transform


def object_bbox(cam: Camera, pose: torch.Tensor, size: torch.Tensor, pad: float = 0.1,
                min_side: int = 16) -> tuple[int, int, int, int] | None:
    """Pixel box (x0, y0, x1, y1) covering a 3D box in ``cam``, or None if it is behind the camera,
    off-screen or smaller than ``min_side`` pixels."""
    corners = torch.from_numpy(box_corners(pose, size))
    w2c = torch.linalg.inv(cam.c2w.float())
    pc = corners @ w2c[:3, :3].T + w2c[:3, 3]
    if (pc[:, 2] < 0.3).any():
        return None
    uv = pc[:, :2] / pc[:, 2:3] * torch.stack([cam.K[0, 0], cam.K[1, 1]]) + cam.K[:2, 2]
    lo, hi = uv.min(0).values, uv.max(0).values
    ext = (hi - lo) * pad
    x0, y0 = [int(max(0, float(v))) for v in lo - ext]
    x1 = int(min(cam.width, float(hi[0] + ext[0])))
    y1 = int(min(cam.height, float(hi[1] + ext[1])))
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None
    return x0, y0, x1, y1


def object_boxes(graph, cam: Camera) -> list[tuple[int, int, int, int]]:
    boxes = []
    for node in graph.objects.values():
        if not node.is_visible(cam.frame_idx):
            continue
        rot, trans = node.pose(cam.frame_idx)
        bb = object_bbox(cam, make_transform(rot.detach().cpu(), trans.detach().cpu()), node.size.cpu())
        if bb is not None:
            boxes.append(bb)
    return boxes


@torch.no_grad()
def evaluate_views(graph, views, backend: str, device, sh_degree: int | None = None, use_lpips: bool = True) -> dict:
    """Image metrics over whole frames, plus ``obj_*`` metrics over the dynamic objects' image boxes.

    Whole-frame numbers are dominated by easy regions (sky, road), so object quality is reported
    separately."""
    lp = LPIPS() if use_lpips and LPIPS.available() else None
    acc: dict[str, list[float]] = {"psnr": [], "ssim": [], "obj_psnr": [], "obj_ssim": []}
    if lp is not None:
        acc["lpips"] = []
    for cam, img in views:
        gt = img.to(device).float() / 255.0 if img.dtype == torch.uint8 else img.to(device)
        pred = graph.render(cam, sh_degree, backend)["rgb"].clamp(0, 1)
        acc["psnr"].append(float(psnr(pred, gt)))
        acc["ssim"].append(float(ssim(pred, gt)))
        if lp is not None:
            acc["lpips"].append(float(lp(pred, gt)))
        for x0, y0, x1, y1 in object_boxes(graph, cam):
            acc["obj_psnr"].append(float(psnr(pred[y0:y1, x0:x1], gt[y0:y1, x0:x1])))
            acc["obj_ssim"].append(float(ssim(pred[y0:y1, x0:x1], gt[y0:y1, x0:x1])))
    out = {k: float(np.mean(v)) for k, v in acc.items() if v}
    out["num_views"] = len(views)
    out["num_object_crops"] = len(acc["obj_psnr"])
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
