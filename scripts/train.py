#!/usr/bin/env python
"""Train a scene graph.

    python scripts/train.py --config configs/synthetic.yaml
    python scripts/train.py --config configs/nuscenes_mini.yaml --scene scene-0061 train.max_steps=7000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nr.train.trainer import Trainer  # noqa: E402
from nr.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scene", default=None, help="shortcut for data.cache_path=data/cache/<scene>.pt")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None, help="output directory (default: <output_dir>/<scene name>)")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. train.ssim_lambda=0.1")
    args = ap.parse_args()

    overrides = list(args.overrides)
    if args.scene:
        overrides.append(f"data.cache_path=data/cache/{args.scene}.pt")
    if args.max_steps:
        overrides.append(f"train.max_steps={args.max_steps}")
    cfg = load_config(args.config, overrides)
    trainer = Trainer(cfg, out_dir=args.out)
    print(
        f"scene {trainer.scene.name}: {len(trainer.train_views)} train / {len(trainer.test_views)} test views, "
        f"{trainer.graph.num_gaussians()} Gaussians, {len(trainer.graph.objects)} objects, "
        f"backend={trainer.backend}, device={trainer.device}"
    )
    trainer.train()
    print(f"checkpoint: {trainer.out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
