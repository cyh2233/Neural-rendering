#!/usr/bin/env python
"""Export the scene at one frame (background + objects placed at that frame) as a standard 3DGS PLY.

    python scripts/export_ply.py --ckpt outputs/scene-0061/last.pt --frame 0 --out scene_f0.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nr.scene.gaussians import save_ply  # noqa: E402
from nr.train.trainer import load_checkpoint  # noqa: E402
from nr.utils.geometry import quat_multiply, rotmat_to_quat, yaw_to_quat  # noqa: E402


@torch.no_grad()
def export_frame(graph, frame: int, path: str | Path, background: bool = True, objects: bool = True) -> int:
    parts = []
    if background:
        g = graph.background.gaussians
        parts.append((g.means, g.quats, g))
    if objects:
        for node in graph.objects.values():
            if not node.is_visible(frame):
                continue
            g = node.gaussians
            rot, trans = node.pose(frame)
            q = quat_multiply(rotmat_to_quat(node.base_poses[frame, :3, :3]), yaw_to_quat(node.delta_yaw[frame]))
            parts.append((g.means @ rot.T + trans, quat_multiply(q.expand_as(g.quats), g.quats), g))
    save_ply(
        path,
        torch.cat([m for m, _, _ in parts]),
        torch.cat([q for _, q, _ in parts]),
        torch.cat([g.scales for _, _, g in parts]),
        torch.cat([g.opacities for _, _, g in parts]),
        torch.cat([g.sh0 for _, _, g in parts]),
        torch.cat([g.shN for _, _, g in parts]),
    )
    return sum(p[0].shape[0] for p in parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-objects", action="store_true")
    ap.add_argument("--no-background", action="store_true")
    args = ap.parse_args()
    graph, _, _ = load_checkpoint(args.ckpt)
    out = args.out or str(Path(args.ckpt).parent / f"frame_{args.frame:04d}.ply")
    n = export_frame(graph, args.frame, out, not args.no_background, not args.no_objects)
    print(f"wrote {n} Gaussians to {out}")


if __name__ == "__main__":
    main()
