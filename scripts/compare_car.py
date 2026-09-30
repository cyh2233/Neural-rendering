#!/usr/bin/env python
"""Side-by-side comparison of held-out views, focused on the dynamic objects.

Writes two images next to the checkpoint (or to --out):
  <out>_frames.png : per held-out view  [ground truth | render | background only | shifted view]
  <out>_crops.png  : per visible object  [ground truth crop | render crop], enlarged

    python scripts/compare_car.py --ckpt outputs/synthetic_car/last.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nr.data.loader import load_scene_from_config  # noqa: E402
from nr.eval.nvs import evaluate_views, object_boxes  # noqa: E402
from nr.render.backend import resolve_backend  # noqa: E402
from nr.train.trainer import load_checkpoint, prepare_views, resolve_device  # noqa: E402
from nr.utils.geometry import lateral_shift  # noqa: E402


def to_u8(x: torch.Tensor) -> np.ndarray:
    return (x.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


def enlarge(img: np.ndarray, height: int) -> np.ndarray:
    k = max(1, height // img.shape[0])
    return img.repeat(k, 0).repeat(k, 1)


def pad_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.full((h, w, 3), 255, np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="output prefix (default: <ckpt dir>/compare)")
    ap.add_argument("--lateral", type=float, default=-1.5, help="shift for the 4th column [m], + = camera right")
    ap.add_argument("--max-views", type=int, default=6)
    ap.add_argument("--crop-height", type=int, default=160)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()

    import imageio.v3 as iio

    device = resolve_device(args.device)
    backend = resolve_backend(args.backend)
    graph, cfg, _ = load_checkpoint(args.ckpt, device)
    scene = load_scene_from_config(cfg)
    _, test_cams = scene.split(cfg.data.holdout_every)
    views = prepare_views(test_cams, cfg.data.image_scale)
    prefix = Path(args.out) if args.out else Path(args.ckpt).parent / "compare"

    metrics = evaluate_views(graph, views, backend, device)
    print(json.dumps(metrics, indent=2))

    with_obj = [(c, im) for c, im in views if object_boxes(graph, c)]
    chosen = with_obj[: args.max_views] or views[: args.max_views]
    rows, crops = [], []
    for cam, img in chosen:
        gt = img.numpy() if img.dtype == torch.uint8 else to_u8(img)
        full = to_u8(graph.render(cam, backend=backend)["rgb"])
        bg = to_u8(graph.render(cam, backend=backend, include_objects=False)["rgb"])
        shifted = to_u8(graph.render(cam.with_pose(lateral_shift(cam.c2w, args.lateral)), backend=backend)["rgb"])
        rows.append(np.concatenate([gt, full, bg, shifted], 1))
        for x0, y0, x1, y1 in object_boxes(graph, cam):
            pair = np.concatenate([gt[y0:y1, x0:x1], np.full((y1 - y0, 2, 3), 255, np.uint8), full[y0:y1, x0:x1]], 1)
            crops.append(enlarge(pair, args.crop_height))

    sep = np.full((4, rows[0].shape[1], 3), 255, np.uint8)
    frames = np.concatenate([x for r in rows for x in (r, sep)][:-1], 0)
    iio.imwrite(f"{prefix}_frames.png", enlarge(frames, 2 * frames.shape[0]) if frames.shape[0] < 400 else frames)
    print(f"wrote {prefix}_frames.png")
    if crops:
        w = max(c.shape[1] for c in crops)
        h = max(c.shape[0] for c in crops)
        iio.imwrite(f"{prefix}_crops.png", np.concatenate([pad_to(c, h + 6, w) for c in crops], 0))
        print(f"wrote {prefix}_crops.png")


if __name__ == "__main__":
    main()
