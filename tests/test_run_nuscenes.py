import subprocess
import sys

from nr.data.cache import save_scene
from nr.data.synthetic_car import make_car_scene
from nr.utils.preflight import FAIL, OK, check_nuscenes


def test_check_nuscenes_layout(tmp_path):
    root = tmp_path / "nuscenes"
    checks = {c.name: c for c in check_nuscenes(root)}
    assert checks["dataroot"].status == FAIL

    (root / "v1.0-mini").mkdir(parents=True)
    checks = {c.name: c for c in check_nuscenes(root)}
    assert checks["metadata"].status == FAIL and "scene.json" in checks["metadata"].detail
    assert checks["samples/CAM_FRONT"].status == FAIL

    for t in ("scene", "sample", "sample_data", "calibrated_sensor", "ego_pose", "sample_annotation"):
        (root / "v1.0-mini" / f"{t}.json").write_text("[]")
    (root / "samples" / "CAM_FRONT").mkdir(parents=True)
    (root / "samples" / "CAM_FRONT" / "a.jpg").write_bytes(b"x")
    (root / "samples" / "LIDAR_TOP").mkdir(parents=True)
    (root / "samples" / "LIDAR_TOP" / "a.pcd.bin").write_bytes(b"x")
    checks = {c.name: c for c in check_nuscenes(root)}
    for name in ("metadata", "samples/CAM_FRONT", "samples/LIDAR_TOP"):
        assert checks[name].status == OK, checks[name]


def test_run_nuscenes_end_to_end_on_cpu(tmp_path):
    """The one-click runner on a small cached scene, without GPU or real data."""
    cache = tmp_path / "cache.pt"
    save_scene(make_car_scene(num_frames=8, width=96, height=54, depth_samples=50), cache)
    out = tmp_path / "run"
    r = subprocess.run(
        [sys.executable, "scripts/run_nuscenes.py", "--cache", str(cache), "--allow-cpu", "--steps", "6",
         "--out", str(out), "data.image_scale=1.0", "train.log_every=3", "train.eval_every=0",
         "train.save_every=0", "model.max_bg_points=3000", "model.object_points_per_box=300",
         "strategy.enabled=false"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    report = (out / "report.md").read_text()
    assert "Held-out metrics" in report and "| psnr |" in report
    assert (out / "last.pt").exists() and (out / "metrics.json").exists()
    assert (out / "compare_frames.png").exists()
    video = [p for p in out.iterdir() if p.name.startswith("front_trajectory")]
    assert video, list(out.iterdir())
