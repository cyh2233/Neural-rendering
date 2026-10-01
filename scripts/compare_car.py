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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nr.data.loader import load_scene_from_config  # noqa: E402
from nr.eval.compare import write_comparisons  # noqa: E402
from nr.eval.nvs import evaluate_views  # noqa: E402
from nr.render.backend import resolve_backend  # noqa: E402
from nr.train.trainer import load_checkpoint, prepare_views, resolve_device  # noqa: E402


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

    device = resolve_device(args.device)
    backend = resolve_backend(args.backend)
    graph, cfg, _ = load_checkpoint(args.ckpt, device)
    scene = load_scene_from_config(cfg)
    _, test_cams = scene.split(cfg.data.holdout_every)
    views = prepare_views(test_cams, cfg.data.image_scale)
    prefix = Path(args.out) if args.out else Path(args.ckpt).parent / "compare"
    print(json.dumps(evaluate_views(graph, views, backend, device), indent=2))
    for path in write_comparisons(graph, views, backend, prefix, args.lateral, args.max_views, args.crop_height):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
