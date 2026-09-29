#!/usr/bin/env python
"""Interactive browser viewer for a trained checkpoint (requires ``pip install -e ".[viewer]"``).

    python scripts/view.py --ckpt outputs/scene-0061/last.pt --port 8080

On a remote GPU machine, forward the port: ``ssh -L 8080:localhost:8080 <host>``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nr.data.loader import load_scene_from_config  # noqa: E402
from nr.render.backend import resolve_backend  # noqa: E402
from nr.train.trainer import load_checkpoint, resolve_device  # noqa: E402
from nr.viz.viewer import SceneViewer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=None, help="initial render width in pixels")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    graph, cfg, _ = load_checkpoint(args.ckpt, device)
    scene = load_scene_from_config(cfg)
    backend = resolve_backend(args.backend)
    width = args.width or (640 if backend == "gsplat" else 160)  # the torch rasterizer is slow
    viewer = SceneViewer(graph, scene, backend=backend, host=args.host, port=args.port, width=width)
    print(f"backend={backend} device={device} objects={len(graph.objects)} frames={scene.num_frames}")
    viewer.run()


if __name__ == "__main__":
    main()
