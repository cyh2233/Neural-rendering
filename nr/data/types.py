"""Dataset-agnostic scene description.

Every data source (nuScenes, synthetic, later Waymo / KITTI-360) converts into a
:class:`SceneData`. The rest of the pipeline only ever sees these types.

World frame: right-handed, z-up, origin at the first ego pose of the log.
Camera frame: OpenCV (+x right, +y down, +z forward).
Box frame: +x forward (length), +y left (width), +z up (height), origin at box centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Camera:
    K: torch.Tensor                 # (3, 3) intrinsics in pixels
    c2w: torch.Tensor               # (4, 4) camera-to-world
    width: int
    height: int
    cam_id: int                     # sensor index, e.g. 0..5 for nuScenes surround cameras
    frame_idx: int                  # index into SceneData.frame_timestamps
    image_path: str | None = None
    image: torch.Tensor | None = None       # (H, W, 3) uint8 or float in [0, 1]
    lidar_uvz: torch.Tensor | None = None   # (M, 3) projected LiDAR: pixel u, pixel v, depth [m]
    name: str = ""

    @property
    def viewmat(self) -> torch.Tensor:
        from nr.utils.geometry import invert_transform

        return invert_transform(self.c2w)

    @property
    def center(self) -> torch.Tensor:
        return self.c2w[:3, 3]

    def scaled(self, s: float) -> Camera:
        """Return a copy whose intrinsics / resolution are scaled by ``s`` (image is not resized)."""
        if s == 1.0:
            return self
        K = self.K.clone()
        K[:2] *= s
        lidar = None
        if self.lidar_uvz is not None:
            lidar = self.lidar_uvz.clone()
            lidar[:, :2] *= s
        return Camera(
            K=K, c2w=self.c2w, width=int(round(self.width * s)), height=int(round(self.height * s)),
            cam_id=self.cam_id, frame_idx=self.frame_idx, image_path=self.image_path,
            image=None, lidar_uvz=lidar, name=self.name,
        )

    def with_pose(self, c2w: torch.Tensor) -> Camera:
        return Camera(
            K=self.K, c2w=c2w, width=self.width, height=self.height, cam_id=self.cam_id,
            frame_idx=self.frame_idx, image_path=None, image=None, lidar_uvz=None, name=self.name + "_novel",
        )


@dataclass
class BoxTrack:
    """A rigid object tracked through the log."""

    instance_id: str
    category: str
    size: torch.Tensor          # (3,) length (x), width (y), height (z) in metres
    poses: torch.Tensor         # (T, 4, 4) box-to-world for every frame of the scene
    valid: torch.Tensor         # (T,) bool: object annotated in that frame
    points: torch.Tensor | None = None   # (M, 3) LiDAR points in box-local coordinates
    colors: torch.Tensor | None = None   # (M, 3) in [0, 1]


@dataclass
class SceneData:
    name: str
    cameras: list[Camera]
    frame_timestamps: np.ndarray            # (T,) seconds
    points: torch.Tensor                    # (N, 3) background LiDAR points (world)
    colors: torch.Tensor | None = None      # (N, 3) in [0, 1]
    tracks: list[BoxTrack] = field(default_factory=list)
    sensor_names: list[str] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frame_timestamps)

    @property
    def num_sensors(self) -> int:
        return max(len(self.sensor_names), 1 + max((c.cam_id for c in self.cameras), default=0))

    def split(self, holdout_every: int) -> tuple[list[Camera], list[Camera]]:
        """Hold out whole frames: frame f is a test frame if f % N == N - 1."""
        if holdout_every <= 0:
            return list(self.cameras), []
        train = [c for c in self.cameras if c.frame_idx % holdout_every != holdout_every - 1]
        test = [c for c in self.cameras if c.frame_idx % holdout_every == holdout_every - 1]
        return train, test
