"""Save / load a SceneData to a single ``.pt`` file.

Images are stored by path (not pixels) so caches stay small; they are loaded lazily
by :func:`load_image`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from nr.data.types import BoxTrack, Camera, SceneData


def save_scene(scene: SceneData, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cams = []
    for c in scene.cameras:
        d = dataclasses.asdict(c)
        if c.image_path is not None:
            d["image"] = None  # reload from disk
        cams.append(d)
    obj = {
        "name": scene.name,
        "cameras": cams,
        "frame_timestamps": torch.from_numpy(np.asarray(scene.frame_timestamps)),
        "points": scene.points,
        "colors": scene.colors,
        "tracks": [dataclasses.asdict(t) for t in scene.tracks],
        "sensor_names": list(scene.sensor_names),
    }
    torch.save(obj, path)


def load_scene(path: str | Path) -> SceneData:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return SceneData(
        name=obj["name"],
        cameras=[Camera(**c) for c in obj["cameras"]],
        frame_timestamps=obj["frame_timestamps"].numpy(),
        points=obj["points"],
        colors=obj["colors"],
        tracks=[BoxTrack(**t) for t in obj["tracks"]],
        sensor_names=obj["sensor_names"],
    )


def load_image(cam: Camera, scale: float = 1.0) -> torch.Tensor:
    """(H, W, 3) float32 in [0, 1] at the camera's (possibly scaled) resolution."""
    if cam.image is not None:
        img = cam.image
    else:
        import imageio.v3 as iio

        img = torch.from_numpy(np.asarray(iio.imread(cam.image_path))[..., :3].copy())
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    h, w = int(round(img.shape[0] * scale)), int(round(img.shape[1] * scale))
    if (h, w) != tuple(img.shape[:2]):
        img = torch.nn.functional.interpolate(
            img.permute(2, 0, 1)[None], size=(h, w), mode="area"
        )[0].permute(1, 2, 0)
    return img.contiguous()
