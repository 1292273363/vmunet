import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import load_config_class


CONFIGS = {
    'V1': 'config_setting_isic2018_V1_vssblock_spscan_one_path.py',
    'V2': 'config_setting_isic2018_V2_vssblock_spscan_two_paths.py',
    'V3': 'config_setting_isic2018_V3_vssblock_spscan_two_paths_spatial.py',
}


def _assert_equal(name, actual, expected):
    if actual != expected:
        raise AssertionError(f'{name}: expected {expected!r}, got {actual!r}')


def _assert_true(name, value):
    if not value:
        raise AssertionError(f'{name}: expected True, got {value!r}')


def main():
    seen_output_dirs = set()
    seen_exp_names = set()

    for tag, module_name in CONFIGS.items():
        config = load_config_class(module_name)
        model_cfg = config.model_config
        sp_cfg = getattr(config, 'sp_scan_cfg', None)
        if not isinstance(sp_cfg, dict):
            raise AssertionError(f'{tag}: sp_scan_cfg must be a dict.')

        _assert_equal(f'{tag}.use_sp_rgm', getattr(config, 'use_sp_rgm', None), False)
        _assert_equal(f'{tag}.model_config.use_sp_rgm', model_cfg.get('use_sp_rgm'), False)
        _assert_equal(f'{tag}.use_sp_scan', getattr(config, 'use_sp_scan', None), True)
        _assert_equal(f'{tag}.model_config.use_sp_scan', model_cfg.get('use_sp_scan'), True)
        _assert_equal(f'{tag}.sp_scan_stage', getattr(config, 'sp_scan_stage', None), 'bottleneck_last')
        _assert_equal(f'{tag}.model_config.sp_scan_stage', model_cfg.get('sp_scan_stage'), 'bottleneck_last')
        _assert_equal(f'{tag}.num_regions', tuple(sp_cfg.get('num_regions')), (2, 2))
        _assert_equal(f'{tag}.tau', sp_cfg.get('tau'), 0.2)
        _assert_equal(f'{tag}.xy_weight', sp_cfg.get('xy_weight'), 2.0)
        _assert_equal(f'{tag}.normalize_assign', sp_cfg.get('normalize_assign'), True)
        _assert_equal(f'{tag}.assign_norm', sp_cfg.get('assign_norm'), 'layer')
        _assert_equal(f'{tag}.lambda_region', getattr(config, 'lambda_region', None), 0.0)
        _assert_equal(f'{tag}.lambda_compact', getattr(config, 'lambda_compact', None), 0.0)
        _assert_equal(f'{tag}.lambda_balance', getattr(config, 'lambda_balance', None), 0.0)

        if tag == 'V1':
            _assert_equal('V1.replace_mode', sp_cfg.get('replace_mode'), 'one_path')
            _assert_equal('V1.feat_weight', sp_cfg.get('feat_weight'), 0.1)
        elif tag == 'V2':
            _assert_equal('V2.replace_mode', sp_cfg.get('replace_mode'), 'two_paths')
            _assert_equal('V2.feat_weight', sp_cfg.get('feat_weight'), 0.1)
        elif tag == 'V3':
            _assert_equal('V3.replace_mode', sp_cfg.get('replace_mode'), 'two_paths')
            if sp_cfg.get('feat_weight') not in (0.0, 0.05):
                raise AssertionError(f"V3.feat_weight must be 0.0 or 0.05, got {sp_cfg.get('feat_weight')!r}")

        _assert_true(f'{tag}.output_dir unique', config.output_dir not in seen_output_dirs)
        _assert_true(f'{tag}.exp_name unique', config.exp_name not in seen_exp_names)
        seen_output_dirs.add(config.output_dir)
        seen_exp_names.add(config.exp_name)

        print(
            f"{tag}: exp_name={config.exp_name}, output_dir={config.output_dir}, "
            f"replace_mode={sp_cfg['replace_mode']}, feat_weight={sp_cfg['feat_weight']}"
        )

    print('VSSBlock SPScan config check passed.')


if __name__ == '__main__':
    main()
