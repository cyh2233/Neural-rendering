import json
import subprocess
import sys

import torch

from nr.train.trainer import Trainer, load_checkpoint
from nr.utils.config import load_config


def _cfg(tmp_path, *extra):
    return load_config(
        "configs/synthetic.yaml",
        [f"output_dir={tmp_path}", "train.log_every=10", "synthetic.num_frames=4",
         "synthetic.width=40", "synthetic.height=30", "data.holdout_every=4", *extra],
    )


def test_training_reduces_loss_and_densifies(tmp_path):
    cfg = _cfg(tmp_path, "train.max_steps=80", "train.depth_lambda=0.05",
               "strategy.refine_start=20", "strategy.refine_every=20", "strategy.refine_stop=70")
    trainer = Trainer(cfg)
    n0 = trainer.graph.num_gaussians()
    hist = trainer.train(verbose=False)
    assert hist[-1]["loss"] < 0.6 * hist[0]["loss"]
    assert hist[-1]["psnr"] > hist[0]["psnr"] + 3
    assert trainer.graph.num_gaussians() != n0  # densification / pruning happened
    metrics = json.loads((trainer.out_dir / "metrics.json").read_text())
    assert metrics["num_views"] == 2 and metrics["psnr"] > 10


def test_checkpoint_reload_renders_identically(tmp_path):
    trainer = Trainer(_cfg(tmp_path, "train.max_steps=15"))
    trainer.train(verbose=False)
    graph, cfg, ckpt = load_checkpoint(trainer.out_dir / "last.pt")
    cam = trainer.train_views[0][0]
    with torch.no_grad():
        a = trainer.graph.render(cam, backend="torch")["rgb"]
        b = graph.render(cam, backend="torch")["rgb"]
    assert ckpt["step"] == 15
    assert torch.allclose(a, b, atol=1e-6)


def test_scripts_end_to_end(tmp_path):
    out = tmp_path / "run"
    base = [sys.executable]
    over = ["synthetic.num_frames=4", "synthetic.width=32", "synthetic.height=24", "data.holdout_every=4"]
    subprocess.run(base + ["scripts/train.py", "--config", "configs/synthetic.yaml", "--max-steps", "10",
                           "--out", str(out), *over], check=True, capture_output=True)
    ckpt = out / "last.pt"
    r = subprocess.run(base + ["scripts/render.py", "--ckpt", str(ckpt), "--mode", "heldout"],
                       check=True, capture_output=True, text=True)
    assert '"psnr"' in r.stdout
    subprocess.run(base + ["scripts/render.py", "--ckpt", str(ckpt), "--mode", "shift", "--lateral", "1.0",
                           "--out", str(tmp_path / "shift")], check=True, capture_output=True)
    assert len(list((tmp_path / "shift").glob("*.png"))) == 4
    ply = tmp_path / "f0.ply"
    subprocess.run(base + ["scripts/export_ply.py", "--ckpt", str(ckpt), "--out", str(ply)],
                   check=True, capture_output=True)
    assert ply.stat().st_size > 1000
