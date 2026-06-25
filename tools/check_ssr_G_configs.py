import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.vmunet.vmunet import VMUNet
from utils import load_config_class


CONFIGS = {
    'G1': 'config_setting_isic2018_G1_ssr_stage2_skip_avgmax.py',
    'G2': 'config_setting_isic2018_G2_ssr_stage1_skip_avgmax.py',
}


def _assert_equal(name, actual, expected):
    if actual != expected:
        raise AssertionError(f'{name}: expected {expected!r}, got {actual!r}')


def _build_model(config, use_ssr=None):
    model_cfg = config.model_config
    if use_ssr is None:
        use_ssr = model_cfg.get('use_ssr', False)
    return VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=None,
        use_sp_rgm=False,
        sp_rgm_cfg=None,
        use_sp_scan=False,
        sp_scan_cfg=None,
        sp_scan_stage=None,
        sp_scan_blocks=None,
        use_ssr=use_ssr,
        ssr_cfg=model_cfg.get('ssr_cfg') if use_ssr else None,
    )


def _non_ssr_state_shapes(model):
    return {
        name: tuple(value.shape)
        for name, value in model.vmunet.state_dict().items()
        if 'ssr_modules' not in name
    }


def _ssr_state_names(model):
    return [name for name in model.vmunet.state_dict() if 'ssr_modules' in name]


def _extract_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'model' in checkpoint:
            return checkpoint['model']
    return checkpoint


def _safe_load_summary(vssm, pretrained_dict):
    model_dict = vssm.state_dict()
    loadable_dict = {}
    unexpected_keys = []
    shape_mismatch = []

    for key, value in pretrained_dict.items():
        if key not in model_dict:
            unexpected_keys.append(key)
            continue
        if model_dict[key].shape != value.shape:
            shape_mismatch.append((key, tuple(value.shape), tuple(model_dict[key].shape)))
            continue
        loadable_dict[key] = value

    load_result = vssm.load_state_dict(loadable_dict, strict=False)
    return {
        'loaded': len(loadable_dict),
        'missing': tuple(load_result.missing_keys),
        'unexpected': tuple(unexpected_keys),
        'shape_mismatch': tuple(shape_mismatch),
    }


def _decoder_pretrained_dict(pretrained_odict):
    pretrained_dict = {}
    for key, value in pretrained_odict.items():
        if 'layers.0' in key:
            pretrained_dict[key.replace('layers.0', 'layers_up.3')] = value
        elif 'layers.1' in key:
            pretrained_dict[key.replace('layers.1', 'layers_up.2')] = value
        elif 'layers.2' in key:
            pretrained_dict[key.replace('layers.2', 'layers_up.1')] = value
        elif 'layers.3' in key:
            pretrained_dict[key.replace('layers.3', 'layers_up.0')] = value
    return pretrained_dict


def _check_pretrained_load(tag, config, model, baseline):
    ckpt_path = config.model_config.get('load_ckpt_path')
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f'{tag}: skipped pretrained load check, checkpoint not found: {ckpt_path}')
        return

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    pretrained_odict = _extract_checkpoint_state(checkpoint)
    for label, pretrained_dict in (
        ('encoder', pretrained_odict),
        ('decoder', _decoder_pretrained_dict(pretrained_odict)),
    ):
        ssr_model = _build_model(config, use_ssr=True)
        base_model = _build_model(config, use_ssr=False)
        ssr_summary = _safe_load_summary(ssr_model.vmunet, pretrained_dict)
        base_summary = _safe_load_summary(base_model.vmunet, pretrained_dict)
        if ssr_summary['shape_mismatch']:
            raise AssertionError(f"{tag}.{label}: shape_mismatch must be 0, got {ssr_summary['shape_mismatch'][:5]}")
        extra_missing = set(ssr_summary['missing']) - set(base_summary['missing'])
        if not extra_missing or not all('ssr_modules' in key for key in extra_missing):
            raise AssertionError(
                f'{tag}.{label}: SSR extra missing keys must be limited to ssr_modules, '
                f'got {sorted(extra_missing)[:10]}'
            )
        print(
            f"{tag}.{label}: loaded={ssr_summary['loaded']}, missing={len(ssr_summary['missing'])}, "
            f"unexpected={len(ssr_summary['unexpected'])}, shape_mismatch={len(ssr_summary['shape_mismatch'])}, "
            f"extra_ssr_missing={len(extra_missing)}"
        )


def main():
    seen_exp_names = set()
    seen_output_dirs = set()

    for tag, module_name in CONFIGS.items():
        config = load_config_class(module_name)
        model_cfg = config.model_config
        ssr_cfg = config.ssr_cfg

        _assert_equal(f'{tag}.use_ssr', config.use_ssr, True)
        _assert_equal(f'{tag}.model_config.use_ssr', model_cfg.get('use_ssr'), True)
        _assert_equal(f'{tag}.use_sp_scan', config.use_sp_scan, False)
        _assert_equal(f'{tag}.model_config.use_sp_scan', model_cfg.get('use_sp_scan'), False)
        _assert_equal(f'{tag}.use_sp_rgm', config.use_sp_rgm, False)
        _assert_equal(f'{tag}.model_config.use_sp_rgm', model_cfg.get('use_sp_rgm'), False)
        _assert_equal(f'{tag}.use_avg_pool', ssr_cfg.get('use_avg_pool'), True)
        _assert_equal(f'{tag}.use_max_pool', ssr_cfg.get('use_max_pool'), True)
        _assert_equal(f'{tag}.use_pos_embed', ssr_cfg.get('use_pos_embed'), True)
        _assert_equal(f'{tag}.use_graph', ssr_cfg.get('use_graph'), False)
        _assert_equal(f'{tag}.region_update', ssr_cfg.get('region_update'), 'mlp')
        _assert_equal(f'{tag}.gamma_init', ssr_cfg.get('gamma_init'), 1e-3)
        _assert_equal(f'{tag}.gate_type', ssr_cfg.get('gate_type'), 'bounded_tanh')
        _assert_equal(f'{tag}.gate_scale', ssr_cfg.get('gate_scale'), 0.1)
        _assert_equal(f'{tag}.norm_type', ssr_cfg.get('norm_type'), 'group')
        _assert_equal(f'{tag}.num_groups', ssr_cfg.get('num_groups'), 8)
        if 'detach_assignment' not in ssr_cfg:
            raise AssertionError(f'{tag}: ssr_cfg must define detach_assignment.')
        _assert_equal(f'{tag}.lambda_region', config.lambda_region, 0.0)
        _assert_equal(f'{tag}.lambda_compact', config.lambda_compact, 0.0)
        _assert_equal(f'{tag}.lambda_balance', config.lambda_balance, 0.0)

        expected_stage = 'stage2' if tag == 'G1' else 'stage1'
        expected_regions = (4, 4) if tag == 'G1' else (8, 8)
        _assert_equal(f'{tag}.ssr_stages', ssr_cfg.get('ssr_stages'), [expected_stage])
        _assert_equal(f'{tag}.{expected_stage}.num_regions', tuple(ssr_cfg['num_regions'][expected_stage]), expected_regions)

        if config.exp_name in seen_exp_names or config.output_dir in seen_output_dirs:
            raise AssertionError(f'{tag}: exp_name or output_dir is not unique.')
        seen_exp_names.add(config.exp_name)
        seen_output_dirs.add(config.output_dir)

        model = _build_model(config, use_ssr=True)
        baseline = _build_model(config, use_ssr=False)
        _assert_equal(f'{tag}.ssr_module_keys', tuple(model.vmunet.ssr_modules.keys()), (expected_stage,))
        _assert_equal(f'{tag}.baseline_ssr_module_keys', tuple(baseline.vmunet.ssr_modules.keys()), tuple())
        _assert_equal(f'{tag}.non_ssr_state_shapes', _non_ssr_state_shapes(model), _non_ssr_state_shapes(baseline))
        _assert_equal(f'{tag}.baseline_get_ssr_stats', baseline.get_ssr_stats(), None)

        new_state_keys = set(model.vmunet.state_dict()) - set(baseline.vmunet.state_dict())
        if not new_state_keys or not all('ssr_modules' in key for key in new_state_keys):
            raise AssertionError(f'{tag}: new state keys must be limited to ssr_modules, got {sorted(new_state_keys)[:10]}')

        channels = model_cfg['depths'] and model.vmunet.dims[int(expected_stage.replace('stage', ''))]
        spatial = 16 if expected_stage == 'stage2' else 32
        module = model.vmunet.ssr_modules[expected_stage].eval()
        with torch.no_grad():
            x = torch.randn(1, channels, spatial, spatial)
            out, stats = module(x, return_stats=True)
        _assert_equal(f'{tag}.ssr_output_shape', tuple(out.shape), tuple(x.shape))
        _assert_equal(f'{tag}.stats.norm_type', stats['norm_type'], 'group')
        _assert_equal(f'{tag}.stats.num_groups', stats['num_groups'], 8)
        _assert_equal(f'{tag}.stats.detach_assignment', stats['detach_assignment'], ssr_cfg['detach_assignment'])

        _check_pretrained_load(tag, config, model, baseline)

        print(
            f"{tag}: ssr_stage={expected_stage}, num_regions={expected_regions}, "
            f"ssr_state_keys={len(_ssr_state_names(model))}, "
            f"ssr_params={model.get_ssr_param_stats()['ssr_params']}, "
            f"total_params={model.get_ssr_param_stats()['total_params']}, "
            f"ssr_params_ratio={model.get_ssr_param_stats()['ssr_params_ratio']:.6f}, "
            f"output_dir={config.output_dir}"
        )

    print('SSR G config check passed.')


if __name__ == '__main__':
    main()
