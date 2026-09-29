import torch

from nr.scene.gaussians import GaussianModel
from nr.train.strategy import duplicate, remove, reset_opacity, split


def _model_and_opt(n=10):
    g = GaussianModel.from_points(torch.randn(n, 3), sh_degree=1)
    opt = torch.optim.Adam([{"params": [p]} for p in g.params().values()])
    # one step so Adam has state
    loss = sum((p**2).sum() for p in g.params().values())
    loss.backward()
    opt.step()
    return g, opt


def _check_consistent(g, opt):
    params = {id(p) for grp in opt.param_groups for p in grp["params"]}
    for p in g.params().values():
        assert id(p) in params
        assert p.shape[0] == len(g)
        st = opt.state[p]
        assert st["exp_avg"].shape == p.shape and st["exp_avg_sq"].shape == p.shape


def test_duplicate_split_remove_keep_optimizer_in_sync():
    g, opt = _model_and_opt(10)
    mask = torch.zeros(10, dtype=torch.bool)
    mask[:3] = True
    duplicate(g, opt, mask)
    assert len(g) == 13
    _check_consistent(g, opt)
    assert torch.equal(g.means[10:], g.means[:3])

    mask = torch.zeros(13, dtype=torch.bool)
    mask[5] = True
    old_scale = g.scales[5].clone()
    split(g, opt, mask)
    assert len(g) == 14  # one removed, two added
    assert torch.allclose(g.scales[-1], old_scale - torch.log(torch.tensor(1.6)))
    _check_consistent(g, opt)

    mask = torch.zeros(14, dtype=torch.bool)
    mask[::2] = True
    remove(g, opt, mask)
    assert len(g) == 7
    _check_consistent(g, opt)

    reset_opacity(g, opt, 0.01)
    assert torch.sigmoid(g.opacities).max() <= 0.01 + 1e-6

    # the optimizer still works after surgery
    loss = sum((p**2).sum() for p in g.params().values())
    loss.backward()
    opt.step()
