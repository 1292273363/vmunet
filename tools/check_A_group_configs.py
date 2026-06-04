import argparse
import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

from models.region_mamba.sp_rgm_block import SuperpixelRegionGraphMambaBlock
from utils import load_config_class, set_seed


A_GROUP_CONFIGS = [
    'config_setting_isic2018_A0_vmunet_baseline.py',
    'config_setting_isic2018_A1_sp_rgm_module_only_4x4.py',
    'config_setting_isic2018_A2_sp_rgm_full_loss_4x4.py',
    'config_setting_isic2018_A3_sp_rgm_single_path_4x4.py',
    'config_setting_isic2018_A4_sp_rgm_multi_path_4x4.py',
]


def _expect(condition, message):
    if not condition:
        raise AssertionError(message)


def _check_config(config_name):
    config = load_config_class(config_name)
    model_cfg = config.model_config
    _expect(
        getattr(config, 'use_sp_rgm', False) == model_cfg.get('use_sp_rgm', False),
        f'{config_name}: top-level use_sp_rgm and model_config["use_sp_rgm"] differ',
    )

    if config_name.startswith('config_setting_isic2018_A0'):
        _expect(config.use_sp_rgm is False, f'{config_name}: A0 must keep use_sp_rgm=False')
        _expect(config.lambda_region == 0.0, f'{config_name}: A0 lambda_region must be 0.0')
        _expect(config.lambda_compact == 0.0, f'{config_name}: A0 lambda_compact must be 0.0')
        sp_cfg = None
    else:
        _expect(config.use_sp_rgm is True, f'{config_name}: A1-A4 must use SP-RGM')
        _expect('sp_rgm_cfg' in model_cfg, f'{config_name}: missing model_config["sp_rgm_cfg"]')
        sp_cfg = model_cfg['sp_rgm_cfg']
        _expect(sp_cfg is not None, f'{config_name}: sp_rgm_cfg is None')
        _expect('num_iters' in sp_cfg, f'{config_name}: sp_rgm_cfg must use num_iters')
        _expect('slic_iters' not in sp_cfg, f'{config_name}: sp_rgm_cfg must not use slic_iters')
        _expect(tuple(sp_cfg['num_regions']) == (4, 4), f'{config_name}: num_regions must be (4, 4)')
        _expect(sp_cfg['num_iters'] == 5, f'{config_name}: num_iters must be 5')
        _expect(sp_cfg['tau'] == 0.1, f'{config_name}: tau must be 0.1')
        _expect(sp_cfg['xy_weight'] == 2.0, f'{config_name}: xy_weight must be 2.0')
        _expect(sp_cfg['k_spatial'] == 4, f'{config_name}: k_spatial must be 4')
        _expect(sp_cfg['k_feature'] == 4, f'{config_name}: k_feature must be 4')
        _expect(sp_cfg['init_gamma'] == 1e-3, f'{config_name}: init_gamma must be 1e-3')
        _expect(sp_cfg.get('allow_gru_fallback') is False, f'{config_name}: formal configs must not allow GRU fallback')
        _expect('path_modes' in sp_cfg, f'{config_name}: missing path_modes')

        if config_name.startswith('config_setting_isic2018_A1'):
            _expect(config.lambda_region == 0.0, f'{config_name}: A1 lambda_region must be 0.0')
            _expect(config.lambda_compact == 0.0, f'{config_name}: A1 lambda_compact must be 0.0')
        else:
            _expect(config.lambda_region == 0.2, f'{config_name}: lambda_region must be 0.2')
            _expect(config.lambda_compact == 0.001, f'{config_name}: lambda_compact must be 0.001')

        if config_name.startswith('config_setting_isic2018_A3'):
            _expect(sp_cfg['path_modes'] == ['graph'], f'{config_name}: A3 path_modes must be ["graph"]')
        if config_name.startswith('config_setting_isic2018_A4'):
            _expect(
                sp_cfg['path_modes'] == ['yx', 'xy', 'graph', 'reverse_graph'],
                f'{config_name}: A4 must use all four path modes',
            )

    print(
        f"{config_name}: exp_name={config.exp_name}, use_sp_rgm={config.use_sp_rgm}, "
        f"lambda_region={config.lambda_region}, lambda_compact={config.lambda_compact}, "
        f"sp_rgm_cfg={sp_cfg}"
    )


def _run_sp_rgm_smoke():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block = SuperpixelRegionGraphMambaBlock(
        dim=16,
        num_regions=(4, 4),
        num_iters=1,
        tau=0.1,
        xy_weight=2.0,
        k_spatial=4,
        k_feature=4,
        path_modes=['graph'],
        init_gamma=1e-3,
        allow_gru_fallback=True,
    ).to(device)
    block.train()

    x = torch.randn(2, 16, 8, 8, device=device, requires_grad=True)
    out, aux = block(x)
    sp_aux = aux
    _expect(out.shape == x.shape, f'smoke: output shape mismatch {out.shape} vs {x.shape}')
    _expect(sp_aux['Q'].shape == (2, 64, 16), f"smoke: unexpected Q shape {sp_aux['Q'].shape}")
    _expect(sp_aux['path_modes'] == ('graph',), f"smoke: path_modes not applied: {sp_aux['path_modes']}")
    _expect(sp_aux['num_paths_actual'] == 1, f"smoke: num_paths_actual={sp_aux['num_paths_actual']}")

    loss = out.mean() + sp_aux['updated_tokens'].mean() + sp_aux['recon_feat'].mean()
    loss.backward()
    _expect(x.grad is not None, 'smoke: input grad is None')
    _expect(torch.isfinite(x.grad).all().item(), 'smoke: input grad has non-finite values')
    print(
        'SP-RGM smoke passed: '
        f"out={tuple(out.shape)}, Q={tuple(sp_aux['Q'].shape)}, "
        f"uses_mamba={sp_aux['uses_mamba']}, path_modes={sp_aux['path_modes']}"
    )


def main():
    parser = argparse.ArgumentParser(description='Check ISIC2018 A-group configs.')
    parser.add_argument('--smoke', action='store_true', help='Also run SP-RGM block forward/backward smoke test.')
    args = parser.parse_args()

    for config_name in A_GROUP_CONFIGS:
        _check_config(config_name)

    if args.smoke:
        _run_sp_rgm_smoke()


if __name__ == '__main__':
    main()
