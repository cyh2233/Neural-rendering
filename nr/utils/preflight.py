"""Environment checks run before a long GPU job, so problems surface in seconds instead of hours."""

from __future__ import annotations

import importlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

OK, WARN, FAIL = "ok", "warn", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def check_gpu() -> tuple[list[Check], float]:
    """Returns checks and total VRAM in GB (0 if no GPU)."""
    import torch

    if not torch.cuda.is_available():
        return [Check("CUDA GPU", FAIL, f"torch {torch.__version__} sees no CUDA device")], 0.0
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1024**3
    status = OK if vram >= 10 else WARN
    detail = f"{props.name}, {vram:.1f} GB VRAM, torch {torch.__version__}, CUDA {torch.version.cuda}"
    if status == WARN:
        detail += " (under 10 GB: images will be downscaled to 1/4)"
    return [Check("CUDA GPU", status, detail)], vram


def check_gsplat(run_kernel: bool = True) -> list[Check]:
    try:
        gsplat = importlib.import_module("gsplat")
    except ImportError as e:
        return [Check("gsplat", FAIL, f"not installed ({e}); pip install -e '.[gpu]'")]
    if not run_kernel:
        return [Check("gsplat", OK, f"version {gsplat.__version__} (kernels not tested)")]
    import torch

    if not torch.cuda.is_available():
        return [Check("gsplat", FAIL, f"version {gsplat.__version__} installed but no CUDA device")]
    try:
        from nr.render.gsplat_backend import rasterize_gsplat

        t0 = time.time()
        dev = "cuda"
        n = 64
        means = torch.randn(n, 3, device=dev) + torch.tensor([0.0, 0.0, 5.0], device=dev)
        quats = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1)
        scales = torch.full((n, 3), 0.1, device=dev, requires_grad=True)
        opac = torch.full((n,), 0.5, device=dev)
        cols = torch.rand(n, 3, device=dev)
        K = torch.tensor([[50.0, 0, 32], [0, 50, 24], [0, 0, 1]], device=dev)
        rgb, _, _, _ = rasterize_gsplat(means, quats, scales, opac, cols, torch.eye(4, device=dev), K, 64, 48)
        rgb.sum().backward()
        torch.cuda.synchronize()
        return [Check("gsplat", OK, f"version {gsplat.__version__}, forward+backward ran in {time.time() - t0:.1f}s "
                      "(first call includes CUDA kernel compilation)")]
    except Exception as e:  # compilation or runtime failure
        return [Check("gsplat", FAIL, f"version {gsplat.__version__} failed to run: {str(e)[:300]}")]


def check_nuscenes(dataroot: str | Path, version: str = "v1.0-mini") -> list[Check]:
    checks = []
    try:
        importlib.import_module("nuscenes.nuscenes")
        checks.append(Check("nuscenes-devkit", OK, "installed"))
    except ImportError:
        checks.append(Check("nuscenes-devkit", FAIL, "not installed; pip install -e '.[nuscenes]'"))
    root = Path(dataroot)
    if not root.is_dir():
        return checks + [Check("dataroot", FAIL, f"{root} does not exist")]
    tables = root / version
    missing = [t for t in ("scene", "sample", "sample_data", "calibrated_sensor", "ego_pose", "sample_annotation")
               if not (tables / f"{t}.json").is_file()]
    if missing:
        checks.append(Check("metadata", FAIL, f"{tables} is missing {', '.join(m + '.json' for m in missing)}"))
    else:
        checks.append(Check("metadata", OK, f"{tables}"))
    for sensor, pattern in (("CAM_FRONT", "*.jpg"), ("LIDAR_TOP", "*.pcd.bin")):
        d = root / "samples" / sensor
        n = sum(1 for _ in d.glob(pattern)) if d.is_dir() else 0
        checks.append(Check(f"samples/{sensor}", OK if n else FAIL, f"{n} files" if n else f"no {pattern} in {d}"))
    return checks


def check_optional() -> list[Check]:
    out = []
    for mod, name, why in (
        ("imageio_ffmpeg", "ffmpeg (mp4)", "videos fall back to PNG frames; pip install -e '.[video]'"),
        ("lpips", "lpips", "LPIPS metric skipped; pip install -e '.[eval]'"),
    ):
        try:
            importlib.import_module(mod)
            out.append(Check(name, OK, "installed"))
        except ImportError:
            out.append(Check(name, WARN, why))
    return out


def check_disk(path: str | Path, need_gb: float = 5.0) -> list[Check]:
    p = Path(path)
    while not p.exists():
        p = p.parent
    free = shutil.disk_usage(p).free / 1024**3
    return [Check("disk", OK if free >= need_gb else WARN, f"{free:.0f} GB free at {p}")]


def format_checks(checks: list[Check]) -> str:
    w = max(len(c.name) for c in checks)
    return "\n".join(f"  [{c.status:>4}] {c.name:<{w}}  {c.detail}" for c in checks)


def has_failures(checks: list[Check]) -> bool:
    return any(c.status == FAIL for c in checks)
