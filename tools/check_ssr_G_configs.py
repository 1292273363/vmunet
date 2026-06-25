import os
import sys

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
        _assert_equal(f'{tag}.use_graph', ssr_cfg.get('use_graph'), False)
        _assert_equal(f'{tag}.region_update', ssr_cfg.get('region_update'), 'mlp')
        _assert_equal(f'{tag}.gamma_init', ssr_cfg.get('gamma_init'), 1e-3)
        _assert_equal(f'{tag}.gate_type', ssr_cfg.get('gate_type'), 'bounded_tanh')
        _assert_equal(f'{tag}.gate_scale', ssr_cfg.get('gate_scale'), 0.1)
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

        new_state_keys = set(model.vmunet.state_dict()) - set(baseline.vmunet.state_dict())
        if not new_state_keys or not all('ssr_modules' in key for key in new_state_keys):
            raise AssertionError(f'{tag}: new state keys must be limited to ssr_modules, got {sorted(new_state_keys)[:10]}')

        print(
            f"{tag}: ssr_stage={expected_stage}, num_regions={expected_regions}, "
            f"ssr_params={len(_ssr_state_names(model))}, output_dir={config.output_dir}"
        )

    print('SSR G config check passed.')


if __name__ == '__main__':
    main()
