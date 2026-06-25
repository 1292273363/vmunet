import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.region_mamba.superpixel_skip_refine import (  # noqa: E402
    SuperpixelSkipRefine,
    region_hard_max_pool,
    region_soft_avg_pool,
)
from models.vmunet.vmunet import VMUNet  # noqa: E402


def _ssr_cfg(stage, num_regions):
    return {
        'enabled': True,
        'ssr_stages': [stage],
        'num_regions': {stage: num_regions},
        'num_iters': 2,
        'tau': 0.2,
        'xy_weight': 2.0,
        'feat_weight': 0.1,
        'normalize_assign': True,
        'assign_norm': 'layer',
        'use_pos_embed': True,
        'use_avg_pool': True,
        'use_max_pool': True,
        'use_graph': False,
        'region_update': 'mlp',
        'gamma_init': 1e-3,
        'gate_type': 'bounded_tanh',
        'gate_scale': 0.1,
        'debug_stats': True,
    }


def _assert_no_nan(name, tensor):
    if torch.isnan(tensor).any():
        raise AssertionError(f'{name} contains NaN.')


def _test_region_pool_fallback(device):
    feat_flat = torch.randn(2, 16, 8, device=device)
    Q = torch.zeros(2, 16, 4, device=device)
    Q[:, :, 0] = 1.0
    z_avg, _ = region_soft_avg_pool(feat_flat, Q)
    z_max = region_hard_max_pool(feat_flat, Q, z_avg)
    _assert_no_nan('z_avg', z_avg)
    _assert_no_nan('z_max', z_max)
    assert torch.allclose(z_max[:, 1:], z_avg[:, 1:])


def _test_ssr_module(device, label, channels, height, num_regions):
    module = SuperpixelSkipRefine(
        dim=channels,
        num_regions=num_regions,
        num_iters=2,
        tau=0.2,
        xy_weight=2.0,
        feat_weight=0.1,
        normalize_assign=True,
        assign_norm='layer',
        use_pos_embed=True,
        use_avg_pool=True,
        use_max_pool=True,
        use_graph=False,
        region_update='mlp',
        gamma_init=1e-3,
        gate_type='bounded_tanh',
        gate_scale=0.1,
        debug_stats=True,
        stage_name=label,
    ).to(device).train()
    x = torch.randn(2, channels, height, height, device=device)
    out, stats = module(x, return_stats=True)
    assert out.shape == x.shape
    assert stats['num_regions_actual'] == num_regions[0] * num_regions[1]
    assert stats['feat_h'] == height and stats['feat_w'] == height
    assert not bool(stats['z_avg_has_nan'])
    assert not bool(stats['z_max_has_nan'])
    assert abs(float(stats['gate_value'])) <= float(stats['gate_scale']) + 1e-6
    _assert_no_nan('out', out)
    loss = out.mean()
    loss.backward()
    if not any(param.grad is not None for param in module.parameters() if param.requires_grad):
        raise AssertionError(f'{label}: no parameter received gradients.')
    print(f'smoke test passed for {label} standalone SSR')


def _build_model(stage, num_regions):
    return VMUNet(
        input_channels=3,
        num_classes=1,
        depths=[1, 1, 1, 1],
        depths_decoder=[1, 1, 1, 1],
        drop_path_rate=0.0,
        use_sp_rgm=False,
        use_sp_scan=False,
        use_ssr=True,
        ssr_cfg=_ssr_cfg(stage, num_regions),
    )


def _test_baseline_full_model(device):
    model = VMUNet(
        input_channels=3,
        num_classes=1,
        depths=[1, 1, 1, 1],
        depths_decoder=[1, 1, 1, 1],
        drop_path_rate=0.0,
        use_sp_rgm=False,
        use_sp_scan=False,
        use_ssr=False,
    ).to(device).train()
    out = model(torch.randn(1, 3, 64, 64, device=device))
    assert out.shape == (1, 1, 64, 64)
    assert tuple(model.vmunet.ssr_modules.keys()) == tuple()
    out.mean().backward()
    print('smoke test passed for baseline use_ssr=False')


def _test_full_model(device, stage, num_regions, label):
    model = _build_model(stage, num_regions).to(device).train()
    assert tuple(model.vmunet.ssr_modules.keys()) == (stage,)
    out, aux = model(torch.randn(1, 3, 64, 64, device=device), return_aux=True)
    assert out.shape == (1, 1, 64, 64)
    assert 'ssr' in aux
    assert aux['ssr']['ssr_stage'] == stage
    assert aux['ssr']['ssr_enabled_stages'] == (stage,)
    out.mean().backward()
    print(f'smoke test passed for {label}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _test_region_pool_fallback(device)
    _test_ssr_module(device, 'G1 SSR stage2', channels=32, height=16, num_regions=(4, 4))
    _test_ssr_module(device, 'G2 SSR stage1', channels=32, height=32, num_regions=(8, 8))

    if not torch.cuda.is_available():
        print('skip full VM-UNet SSR smoke: VMamba selective_scan requires CUDA.')
        return

    _test_baseline_full_model(device)
    _test_full_model(device, 'stage2', (4, 4), 'G1 SSR stage2')
    _test_full_model(device, 'stage1', (8, 8), 'G2 SSR stage1')


if __name__ == '__main__':
    main()
