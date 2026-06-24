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
        'mode': 'replacement',
        'replace_mode': replace_mode,
        'num_regions': (4, 4),
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
        depths=[1, 1, 2, 1],
        depths_decoder=[1, 1, 1, 1],
        drop_path_rate=0.0,
        use_sp_rgm=False,
        use_sp_scan=use_sp_scan,
        sp_scan_cfg=_sp_scan_cfg(replace_mode) if use_sp_scan else None,
        sp_scan_stage='stage2' if use_sp_scan else None,
        sp_scan_blocks='last' if use_sp_scan else None,
    )


def _enabled_ss2d_indices(model):
    enabled = []
    for stage_idx, layer in enumerate(model.vmunet.layers):
        for block_idx, block in enumerate(layer.blocks):
            if isinstance(block.self_attention, SS2D) and block.self_attention.use_sp_scan:
                enabled.append((stage_idx, block_idx))
    return enabled


def _test_baseline(device):
    model = _build_model(use_sp_scan=False).to(device).train()
    out = model(torch.randn(1, 3, 64, 64, device=device))
    assert out.shape == (1, 1, 64, 64)
    out.mean().backward()
    assert _enabled_ss2d_indices(model) == []
    print('smoke test passed for baseline use_sp_scan=False')


def _test_stage2(device, replace_mode, label):
    model = _build_model(use_sp_scan=True, replace_mode=replace_mode).to(device).train()
    vssm = model.vmunet
    assert vssm.enabled_sp_scan_stage_index == 2
    assert vssm.enabled_sp_scan_block_indices == [vssm.stage2_depth - 1]
    assert _enabled_ss2d_indices(model) == [(2, vssm.stage2_depth - 1)]

    bottleneck_input_hw = {}

    def _record_bottleneck_input(_, inputs):
        bottleneck_input_hw['hw'] = tuple(inputs[0].shape[1:3])

    handle = vssm.layers[3].register_forward_pre_hook(_record_bottleneck_input)
    x = torch.randn(1, 3, 64, 64, device=device)
    try:
        out, aux = model(x, return_aux=True)
    finally:
        handle.remove()

    stats = aux['sp_scan']
    assert out.shape == (1, 1, 64, 64)
    assert stats['sp_scan_stage'] == 'stage2'
    assert stats['enabled_sp_scan_stage_index'] == 2
    assert stats['enabled_sp_scan_block_indices'] == (vssm.stage2_depth - 1,)
    assert stats['replace_mode'] == replace_mode
    assert stats['perm_valid'] is True
    assert stats['uses_mamba'] is True
    assert stats['stage2_feat_h'] > bottleneck_input_hw['hw'][0]
    assert stats['stage2_feat_w'] > bottleneck_input_hw['hw'][1]
    assert stats['token_inner_order'] == 'raster'
    assert stats['graph_order'] == 'greedy'
    out.mean().backward()
    print(f'smoke test passed for {label}')


def main():
    if not torch.cuda.is_available():
        print('skip stage2 F smoke: VMamba selective_scan requires CUDA.')
        return

    device = torch.device('cuda')
    _test_baseline(device)
    _test_stage2(device, 'one_path', 'F1 stage2 one_path')
    _test_stage2(device, 'two_paths', 'F2 stage2 two_paths')


if __name__ == '__main__':
    main()
