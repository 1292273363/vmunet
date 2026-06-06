import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.region_mamba.sp_rgm_block import SuperpixelRegionGraphMambaBlock


def main():
    """Run a minimal smoke test for the SoftSLIC distance-fix path."""
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 96

    block = SuperpixelRegionGraphMambaBlock(
        dim=dim,
        num_regions=(2, 2),
        num_iters=2,
        tau=0.2,
        xy_weight=2.0,
        feat_weight=0.2,
        normalize_assign=True,
        assign_norm='layer',
        return_distance_stats=True,
        k_spatial=3,
        k_feature=3,
        path_modes=['yx', 'xy', 'graph', 'reverse_graph'],
        init_gamma=1e-3,
        allow_gru_fallback=True,
    ).to(device)
    block.train()

    x = torch.randn(2, dim, 8, 8, device=device, requires_grad=True)
    out, aux = block(x)

    assert out.shape == x.shape, f'out shape mismatch: {out.shape} vs {x.shape}'
    assert aux['Q'].shape == (2, 64, 4), f"unexpected Q shape: {aux['Q'].shape}"
    assert aux['region_tokens'].shape == (2, 4, dim), (
        f"unexpected region_tokens shape: {aux['region_tokens'].shape}"
    )

    required_aux_keys = (
        'Q_entropy',
        'q_max_mean',
        'empty_region_ratio',
        'region_mass_cv',
        'dist_margin_mean',
        'logit_margin_mean',
        'feat_dist_mean',
        'xy_dist_mean',
        'feat_xy_dist_ratio',
        'uses_mamba',
        'path_modes',
    )
    for key in required_aux_keys:
        assert key in aux, f'missing aux key: {key}'

    loss = out.mean() + aux['recon_feat'].mean()
    loss.backward()

    grad_params = [
        p for p in block.parameters()
        if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
    ]
    assert grad_params, 'no finite parameter gradients found'

    print(
        'smoke test passed: '
        f'out={tuple(out.shape)}, Q={tuple(aux["Q"].shape)}, '
        f'region_tokens={tuple(aux["region_tokens"].shape)}, '
        f'Q_entropy={aux["Q_entropy"].item():.6f}, '
        f'q_max_mean={aux["q_max_mean"].item():.6f}, '
        f'logit_margin_mean={aux["logit_margin_mean"].item():.6f}, '
        f'feat_xy_dist_ratio={aux["feat_xy_dist_ratio"].item():.6f}, '
        f'uses_mamba={aux["uses_mamba"]}, path_modes={aux["path_modes"]}'
    )


if __name__ == '__main__':
    main()
