#!/usr/bin/env python
"""Convert one nuScenes scene into a cached SceneData file.

    python scripts/prepare_nuscenes.py --dataroot /data/nuscenes --version v1.0-mini --scene scene-0061
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nr.data.cache import save_scene  # noqa: E402
from nr.data.nuscenes import (  # noqa: E402
    CAMERAS,
    convert_scene,
    load_nuscenes,
    scene_name_to_token,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", default="scene-0061", help="scene name (e.g. scene-0061) or token")
    ap.add_argument("--out", default=None, help="default: data/cache/<scene>.pt")
    ap.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    ap.add_argument("--categories", nargs="+", default=["vehicle."],
                    help="category prefixes modelled as dynamic rigid objects")
    ap.add_argument("--moving-threshold", type=float, default=1.0)
    ap.add_argument("--no-color", action="store_true", help="skip colouring LiDAR points from images")
    args = ap.parse_args()

    nusc = load_nuscenes(args.dataroot, args.version)
    image_loader = None
    if not args.no_color:
        import imageio.v3 as iio

        image_loader = iio.imread
    scene = convert_scene(
        nusc, args.scene,
        cameras=tuple(args.cameras),
        categories=tuple(args.categories),
        moving_threshold=args.moving_threshold,
        image_loader=image_loader,
        scene_tokens=scene_name_to_token(nusc),
    )
    out = Path(args.out or f"data/cache/{scene.name}.pt")
    save_scene(scene, out)
    print(
        f"saved {out}: {scene.num_frames} frames, {len(scene.cameras)} images, "
        f"{scene.points.shape[0]} background points, {len(scene.tracks)} moving objects"
    )


if __name__ == "__main__":
    main()
