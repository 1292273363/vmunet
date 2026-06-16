import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.vmunet.vmamba import SS2D
from models.vmunet.vmunet import VMUNet


def _sp_scan_cfg(replace_mode):
    return {
        'enabled': True,
        'replace_mode': replace_mode,
        'num_regions': (2, 2),
        'num_iters': 2,
        'tau': 0.2,
        'xy_weight': 2.0,
        'feat_weight': 0.1,
        'normalize_assign': True,
        'assign_norm': 'layer',
        'k_spatial': 3,
        'k_feature': 3,
        'graph_order': 'greedy',
        'token_inner_order': 'raster',
        'detach_order': True,
        'debug_stats': True,
    }


def _build_model(use_sp_scan, replace_mode='one_path'):
    return VMUNet(
        input_channels=3,
        num_classes=1,
        depths=[1, 1, 1, 2],
        depths_decoder=[1, 1, 1, 1],
        drop_path_rate=0.0,
        use_sp_rgm=False,
        use_sp_scan=use_sp_scan,
        sp_scan_cfg=_sp_scan_cfg(replace_mode) if use_sp_scan else None,
        sp_scan_stage='bottleneck' if use_sp_scan else None,
        sp_scan_blocks='all' if use_sp_scan else None,
    )


def _enabled_sp_scan_modules(model):
    return [
        module for module in model.vmunet.layers[-1].modules()
        if isinstance(module, SS2D) and getattr(module, 'use_sp_scan', False)
    ]


def _test_baseline(device):
    model = _build_model(use_sp_scan=False).to(device).train()
    x = torch.randn(1, 3, 64, 64, device=device)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)
    out.mean().backward()
    assert len(_enabled_sp_scan_modules(model)) == 0
    print('smoke test passed for baseline use_sp_scan=False')


def _test_bottleneck_all(device, replace_mode):
    model = _build_model(use_sp_scan=True, replace_mode=replace_mode).to(device).train()
    enabled_modules = _enabled_sp_scan_modules(model)
    assert model.vmunet.bottleneck_depth == 2
    assert model.vmunet.enabled_sp_scan_block_indices == [0, 1]
    assert len(enabled_modules) == model.vmunet.bottleneck_depth

    x = torch.randn(1, 3, 64, 64, device=device)
    out, aux = model(x, return_aux=True)
    assert out.shape == (1, 1, 64, 64)
    assert 'sp_scan' in aux
    assert aux['sp_scan']['sp_scan_blocks'] == 'all'
    assert aux['sp_scan']['enabled_sp_scan_block_indices'] == (0, 1)
    assert aux['sp_scan']['bottleneck_depth'] == 2
    assert aux['sp_scan']['replace_mode'] == replace_mode
    assert aux['sp_scan']['uses_mamba'] is True
    out.mean().backward()

    label = 'one_path' if replace_mode == 'one_path' else 'two_paths'
    print(f'smoke test passed for bottleneck all {label}')


def main():
    if not torch.cuda.is_available():
        print('skip bottleneck-all SPScan smoke: VMamba selective_scan requires CUDA.')
        return

    device = torch.device('cuda')
    _test_baseline(device)
    _test_bottleneck_all(device, 'one_path')
    _test_bottleneck_all(device, 'two_paths')


if __name__ == '__main__':
    main()
