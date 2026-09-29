#!/usr/bin/env python
"""Render a trained checkpoint.

  --mode heldout : render held-out views, report PSNR / SSIM / LPIPS
  --mode train   : re-render the training views
  --mode shift   : render every frame of one camera with a lateral / vertical offset (novel view)

    python scripts/render.py --ckpt outputs/scene-0061/last.pt --mode heldout
    python scripts/render.py --ckpt outputs/scene-0061/last.pt --mode shift --lateral 1.5 --out shift.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nr.data.loader import load_scene_from_config  # noqa: E402
from nr.eval.nvs import evaluate_views, render_views, save_frames  # noqa: E402
from nr.render.backend import resolve_backend  # noqa: E402
from nr.train.trainer import load_checkpoint, prepare_views, resolve_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["heldout", "train", "shift"], default="heldout")
    ap.add_argument("--camera", type=int, default=0, help="sensor index for --mode shift")
    ap.add_argument("--lateral", type=float, default=0.0, help="metres, + = right of the camera")
    ap.add_argument("--up", type=float, default=0.0, help="metres, + = up")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    graph, cfg, _ = load_checkpoint(args.ckpt, device)
    graph.eval()
    backend = resolve_backend(args.backend)
    scene = load_scene_from_config(cfg)
    train_cams, test_cams = scene.split(cfg.data.holdout_every)
    scale = cfg.data.image_scale
    ckpt_dir = Path(args.ckpt).parent

    if args.mode in ("heldout", "train"):
        cams = test_cams if args.mode == "heldout" else train_cams
        views = prepare_views(cams, scale)
        metrics = evaluate_views(graph, views, backend, device)
        print(json.dumps(metrics, indent=2))
        frames = render_views(graph, [c for c, _ in views], backend)
        out = save_frames(frames, args.out or ckpt_dir / f"render_{args.mode}")
    else:
        cams = sorted((c for c in scene.cameras if c.cam_id == args.camera), key=lambda c: c.frame_idx)
        cams = [c.scaled(scale) for c in cams]
        frames = render_views(graph, cams, backend, lateral=args.lateral, up=args.up)
        out = save_frames(frames, args.out or ckpt_dir / f"shift_{args.lateral:+.1f}m.mp4", fps=args.fps)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
