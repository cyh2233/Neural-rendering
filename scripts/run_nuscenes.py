#!/usr/bin/env python
"""One command from raw nuScenes to results: check, prepare, train, evaluate, compare, render videos.

    python scripts/run_nuscenes.py --dataroot /data/nuscenes --scene scene-0061
    python scripts/run_nuscenes.py --dataroot /data/nuscenes --check-only          # environment only
    python scripts/run_nuscenes.py --dataroot /data/nuscenes --steps 7000          # quick first run

Writes to outputs/<scene>/ : last.pt, metrics.json, compare_frames.png, compare_crops.png,
front_trajectory.mp4 (2x2: ground truth | reconstruction / shifted left | shifted right) and report.md.
Extra arguments of the form key.sub=value override the config, e.g. train.depth_lambda=0.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from nr.utils.config import load_config  # noqa: E402
from nr.utils.preflight import (  # noqa: E402
    check_disk,
    check_gpu,
    check_gsplat,
    check_nuscenes,
    check_optional,
    format_checks,
)


class Stages:
    def __init__(self):
        self.times: list[tuple[str, float]] = []

    def run(self, name: str, fn, *a, **kw):
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        out = fn(*a, **kw)
        dt = time.time() - t0
        self.times.append((name, dt))
        print(f"--- {name} done in {dt / 60:.1f} min", flush=True)
        return out


def prepare(args, cache: Path) -> None:
    from nr.data.cache import save_scene
    from nr.data.nuscenes import convert_scene, load_nuscenes, scene_name_to_token

    if cache.exists() and not args.force_prepare:
        print(f"using existing cache {cache} (--force-prepare to rebuild)")
        return
    import imageio.v3 as iio

    nusc = load_nuscenes(args.dataroot, args.version)
    scene = convert_scene(nusc, args.scene, image_loader=iio.imread, scene_tokens=scene_name_to_token(nusc))
    save_scene(scene, cache)
    print(f"{scene.num_frames} frames, {len(scene.cameras)} images, {scene.points.shape[0]} LiDAR points, "
          f"{len(scene.tracks)} moving vehicles -> {cache}")


def write_report(path: Path, args, cfg, env_checks, stages: Stages, metrics: dict, trainer, files: list[Path]):
    lines = [f"# nuScenes reconstruction: {trainer.scene.name}", ""]
    lines += [f"- Machine: {platform.node()}, Python {platform.python_version()}, torch {torch.__version__}"]
    lines += [f"- Backend: {trainer.backend} on {trainer.device}"]
    lines += [f"- Images: {len(trainer.train_views)} train / {len(trainer.test_views)} held out, "
              f"scale {cfg.data.image_scale}"]
    lines += [f"- Steps: {trainer.step}, final Gaussians: {trainer.graph.num_gaussians():,} "
              f"({len(trainer.graph.objects)} dynamic objects)", ""]
    lines += ["## Environment", "", "```", format_checks(env_checks), "```", ""]
    lines += ["## Time", "", "| Stage | Minutes |", "|---|---:|"]
    lines += [f"| {n} | {t / 60:.1f} |" for n, t in stages.times]
    lines += ["", "## Held-out metrics", "", "| Metric | Value |", "|---|---:|"]
    lines += [f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in metrics.items()]
    lines += ["", "obj_* metrics are computed inside the projected boxes of the dynamic vehicles.", ""]
    lines += ["## Files", ""] + [f"- {f.name}" for f in files]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataroot", default=None, help="nuScenes root containing v1.0-mini/, samples/, sweeps/")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", default="scene-0061")
    ap.add_argument("--config", default="configs/nuscenes_mini.yaml")
    ap.add_argument("--steps", type=int, default=None, help="training steps (config default: 30000)")
    ap.add_argument("--image-scale", default="auto", help="'auto' (0.5, or 0.25 under 10 GB VRAM) or a number")
    ap.add_argument("--out", default=None, help="default: outputs/<scene>")
    ap.add_argument("--cache", default=None, help="use this prepared cache instead of converting nuScenes")
    ap.add_argument("--force-prepare", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="run the environment checks and exit")
    ap.add_argument("--allow-cpu", action="store_true", help="run without CUDA/gsplat (very slow; for testing)")
    ap.add_argument("--video-camera", type=int, default=0, help="sensor index for the trajectory video (0 = front)")
    ap.add_argument("--lateral", type=float, default=1.5, help="lateral shift for novel views [m]")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. train.depth_lambda=0")
    args = ap.parse_args()

    # ------------------------------------------------------------------ 1. environment
    print("=== environment checks ===")
    gpu_checks, vram = check_gpu()
    checks = gpu_checks + check_gsplat(run_kernel=torch.cuda.is_available())
    if args.cache is None:
        if args.dataroot is None:
            ap.error("--dataroot is required unless --cache is given")
        checks += check_nuscenes(args.dataroot, args.version)
    checks += check_optional() + check_disk(args.out or "outputs")
    print(format_checks(checks))
    gpu_failed = any(c.status == "FAIL" and c.name in ("CUDA GPU", "gsplat") for c in checks)
    data_failed = any(c.status == "FAIL" and c.name not in ("CUDA GPU", "gsplat") for c in checks)
    if data_failed or (gpu_failed and not args.allow_cpu):
        print("\nFix the FAIL items above, or pass --allow-cpu to run without a GPU (hours per 1000 steps).")
        sys.exit(1)
    if args.check_only:
        print("\nall required checks passed")
        return

    # ------------------------------------------------------------------ 2. config
    cache = Path(args.cache) if args.cache else Path("data/cache") / f"{args.scene}.pt"
    overrides = ["data.source=cache", f"data.cache_path={cache}", *args.overrides]
    if args.steps:
        overrides.append(f"train.max_steps={args.steps}")
    if gpu_failed:
        overrides += ["device=cpu", "backend=torch"]
    if args.image_scale != "auto":
        overrides.append(f"data.image_scale={float(args.image_scale)}")
    elif 0 < vram < 10:
        overrides.append("data.image_scale=0.25")
    cfg = load_config(args.config, overrides)

    stages = Stages()
    if args.cache is None:
        stages.run("prepare nuScenes", prepare, args, cache)

    # ------------------------------------------------------------------ 3. train
    from nr.train.trainer import Trainer

    def train():
        trainer = Trainer(cfg, out_dir=args.out or Path(cfg.output_dir) / Path(cache).stem)
        print(f"{len(trainer.train_views)} train / {len(trainer.test_views)} held-out images, "
              f"{trainer.graph.num_gaussians():,} initial Gaussians, {len(trainer.graph.objects)} vehicles, "
              f"backend={trainer.backend}, device={trainer.device}")
        trainer.train()
        return trainer

    trainer = stages.run("train + evaluate", train)
    out_dir = trainer.out_dir
    metrics = trainer.final_metrics  # held-out metrics, also written to metrics.json by the trainer
    print(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------------ 4. compare + video
    from nr.eval.compare import write_comparisons, write_trajectory_video

    files = [out_dir / "last.pt", out_dir / "metrics.json"]
    if trainer.test_views:
        files += stages.run("comparison images", write_comparisons, trainer.graph, trainer.test_views,
                            trainer.backend, out_dir / "compare", lateral=-args.lateral)
    cams = [c for c in trainer.scene.cameras if c.cam_id == args.video_camera]
    if cams:
        files.append(stages.run("trajectory video", write_trajectory_video, trainer.graph, cams, trainer.backend,
                                out_dir / "front_trajectory.mp4", cfg.data.image_scale, args.lateral))
    report = out_dir / "report.md"
    write_report(report, args, cfg, checks, stages, metrics, trainer, files)
    print(f"\nreport: {report}")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
