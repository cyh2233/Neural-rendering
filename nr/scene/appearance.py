"""Per-sensor affine colour correction (exposure / white balance differences between cameras).

A single 3x3 matrix + bias per *sensor* (not per image) so held-out frames can use it too.
"""

from __future__ import annotations

import torch
from torch import nn


class AppearanceModel(nn.Module):
    def __init__(self, num_sensors: int):
        super().__init__()
        self.affine = nn.Parameter(torch.eye(3).repeat(num_sensors, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_sensors, 3))

    def forward(self, rgb: torch.Tensor, cam_id: int) -> torch.Tensor:
        return rgb @ self.affine[cam_id].T + self.bias[cam_id]

    def regularizer(self) -> torch.Tensor:
        eye = torch.eye(3, device=self.affine.device)
        return ((self.affine - eye) ** 2).mean() + (self.bias**2).mean()
