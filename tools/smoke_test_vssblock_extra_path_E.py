import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.vmunet.vmamba import SS2D
from models.vmunet.vmunet import VMUNet


def _sp_scan_cfg(extra_path_types):
    return {
        'enabled': True,
        'mode': 'extra_path',
        'extra_path_types': extra_path_types,
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
        'gamma_sp_init': 1e-3,
    }


def _build_model(use_sp_scan, extra_path_types=None):
    return VMUNet(
        input_channels=3,
        num_classes=1,
        depths=[1, 1, 1, 2],
        depths_decoder=[1, 1, 1, 1],
        drop_path_rate=0.0,
        use_sp_rgm=False,
        use_sp_scan=use_sp_scan,
        sp_scan_cfg=_sp_scan_cfg(extra_path_types) if use_sp_scan else None,
        sp_scan_stage='bottleneck' if use_sp_scan else None,
        sp_scan_blocks='last2' if use_sp_scan else None,
    )


def _enabled_ss2d(model):
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
    assert not _enabled_ss2d(model)
    print('smoke test passed for baseline use_sp_scan=False')


def _test_extra_path(device, extra_path_types, label):
    model = _build_model(use_sp_scan=True, extra_path_types=extra_path_types).to(device).train()
    enabled_modules = _enabled_ss2d(model)
    assert model.vmunet.bottleneck_depth == 2
    assert model.vmunet.enabled_sp_scan_block_indices == [0, 1]
    assert len(enabled_modules) == 2

    x = torch.randn(1, 3, 64, 64, device=device)
    out, aux = model(x, return_aux=True)
    stats = aux['sp_scan']
    assert out.shape == (1, 1, 64, 64)
    assert stats['sp_scan_mode'] == 'extra_path'
    assert stats['extra_path_types'] == tuple(extra_path_types)
    assert stats['perm_valid'] is True
    assert stats['uses_mamba'] is True
    assert 'y_original_norm' in stats and 'y_graph_norm' in stats
    assert 'graph_orig_norm_ratio' in stats
    assert 'gamma_graph_block0' in stats and 'gamma_graph_block1' in stats
    assert 'gamma_sp_mean' in stats and 'gamma_sp_abs_mean' in stats
    if 'reverse_graph' in extra_path_types:
        assert 'y_reverse_graph_norm' in stats
        assert 'reverse_graph_orig_norm_ratio' in stats
        assert 'gamma_reverse_graph_block0' in stats
        assert 'gamma_reverse_graph_block1' in stats

    loss = out.mean()
    loss.backward()
    for block_idx, module in enumerate(enabled_modules):
        assert module.gamma_graph is not None and module.gamma_graph.requires_grad
        assert module.gamma_graph.grad is not None, f'block {block_idx} gamma_graph has no gradient'
        if 'reverse_graph' in extra_path_types:
            assert module.gamma_reverse_graph is not None and module.gamma_reverse_graph.requires_grad
            assert module.gamma_reverse_graph.grad is not None, (
                f'block {block_idx} gamma_reverse_graph has no gradient'
            )

    print(f'smoke test passed for {label}')


def main():
    if not torch.cuda.is_available():
        print('skip extra-path E smoke: VMamba selective_scan requires CUDA.')
        return

    device = torch.device('cuda')
    _test_baseline(device)
    _test_extra_path(device, ['graph'], 'E1 extra graph')
    _test_extra_path(device, ['graph', 'reverse_graph'], 'E2 extra graph reverse')


if __name__ == '__main__':
    main()
