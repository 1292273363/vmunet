import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.region_mamba.soft_slic import SoftSLIC
from models.region_mamba.sp_scan_ordering import (
    batched_gather_tokens,
    build_superpixel_graph_token_permutation,
)
from models.vmunet.vmamba import VSSBlock


def _test_permutation_helper(device):
    b, c, h, w = 2, 16, 8, 8
    feat = torch.randn(b, c, h, w, device=device)
    soft_slic = SoftSLIC(
        num_regions=(2, 2),
        num_iters=2,
        tau=0.2,
        xy_weight=2.0,
        feat_weight=0.1,
        normalize_assign=True,
        assign_norm='layer',
        return_distance_stats=True,
    ).to(device)
    perm, inv_perm, stats = build_superpixel_graph_token_permutation(
        feat,
        soft_slic,
        k_spatial=3,
        k_feature=3,
        detach_order=True,
    )

    tokens = feat.flatten(2).transpose(1, 2).contiguous()
    restored = batched_gather_tokens(batched_gather_tokens(tokens, perm), inv_perm)

    assert perm.shape == (b, h * w)
    assert inv_perm.shape == (b, h * w)
    assert torch.allclose(restored, tokens)
    assert stats['perm_valid'] is True
    assert stats['num_regions_actual'] == 4
    print('permutation helper smoke passed.')


def _test_vssblock(device, use_sp_scan):
    sp_scan_cfg = {
        'enabled': True,
        'replace_mode': 'two_paths',
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
    block = VSSBlock(
        hidden_dim=16,
        drop_path=0.0,
        d_state=4,
        use_sp_scan=use_sp_scan,
        sp_scan_cfg=sp_scan_cfg if use_sp_scan else None,
    ).to(device)
    block.train()

    x = torch.randn(2, 8, 8, 16, device=device, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape

    loss = out.mean()
    loss.backward()
    has_grad = any(param.grad is not None for param in block.parameters() if param.requires_grad)
    assert has_grad

    if use_sp_scan:
        stats = block.self_attention.last_sp_scan_stats
        assert stats is not None
        assert stats['replace_mode'] == 'two_paths'
        assert stats['perm_valid'] is True
        assert stats['path_types'] == ('raster', 'transpose', 'graph', 'reverse_graph')
    print(f"VSSBlock smoke passed: use_sp_scan={use_sp_scan}.")


def main():
    cpu = torch.device('cpu')
    _test_permutation_helper(cpu)

    if not torch.cuda.is_available():
        print('skip VSSBlock selective_scan smoke: VMamba selective_scan requires CUDA in this environment.')
        return

    device = torch.device('cuda')
    _test_vssblock(device, use_sp_scan=False)
    _test_vssblock(device, use_sp_scan=True)
    print('VSSBlock SPScan smoke test passed.')


if __name__ == '__main__':
    main()
