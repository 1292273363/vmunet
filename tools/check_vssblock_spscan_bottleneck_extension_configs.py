import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.vmunet.vmamba import SS2D
from models.vmunet.vmunet import VMUNet
from utils import load_config_class


CONFIGS = {
    'V4': 'config_setting_isic2018_V4_vssblock_spscan_bottleneck_all_one_path.py',
    'V5': 'config_setting_isic2018_V5_vssblock_spscan_bottleneck_all_two_paths.py',
}


def _assert_equal(name, actual, expected):
    if actual != expected:
        raise AssertionError(f'{name}: expected {expected!r}, got {actual!r}')


def _build_model(config):
    model_cfg = config.model_config
    return VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=None,
        use_sp_rgm=model_cfg.get('use_sp_rgm', False),
        sp_rgm_cfg=model_cfg.get('sp_rgm_cfg'),
        use_sp_scan=model_cfg.get('use_sp_scan', getattr(config, 'use_sp_scan', False)),
        sp_scan_cfg=model_cfg.get('sp_scan_cfg', getattr(config, 'sp_scan_cfg', None)),
        sp_scan_stage=model_cfg.get('sp_scan_stage', getattr(config, 'sp_scan_stage', None)),
        sp_scan_blocks=model_cfg.get('sp_scan_blocks', getattr(config, 'sp_scan_blocks', None)),
    )


def main():
    seen_exp_names = set()
    seen_output_dirs = set()

    for tag, module_name in CONFIGS.items():
        config = load_config_class(module_name)
        model_cfg = config.model_config
        sp_cfg = config.sp_scan_cfg

        _assert_equal(f'{tag}.use_sp_rgm', config.use_sp_rgm, False)
        _assert_equal(f'{tag}.model_config.use_sp_rgm', model_cfg.get('use_sp_rgm'), False)
        _assert_equal(f'{tag}.use_sp_scan', config.use_sp_scan, True)
        _assert_equal(f'{tag}.model_config.use_sp_scan', model_cfg.get('use_sp_scan'), True)
        _assert_equal(f'{tag}.sp_scan_stage', config.sp_scan_stage, 'bottleneck')
        _assert_equal(f'{tag}.model_config.sp_scan_stage', model_cfg.get('sp_scan_stage'), 'bottleneck')
        _assert_equal(f'{tag}.sp_scan_blocks', config.sp_scan_blocks, 'all')
        _assert_equal(f'{tag}.model_config.sp_scan_blocks', model_cfg.get('sp_scan_blocks'), 'all')
        _assert_equal(f'{tag}.num_regions', tuple(sp_cfg.get('num_regions')), (2, 2))
        _assert_equal(f'{tag}.num_iters', sp_cfg.get('num_iters'), 5)
        _assert_equal(f'{tag}.tau', sp_cfg.get('tau'), 0.2)
        _assert_equal(f'{tag}.xy_weight', sp_cfg.get('xy_weight'), 2.0)
        _assert_equal(f'{tag}.feat_weight', sp_cfg.get('feat_weight'), 0.1)
        _assert_equal(f'{tag}.normalize_assign', sp_cfg.get('normalize_assign'), True)
        _assert_equal(f'{tag}.assign_norm', sp_cfg.get('assign_norm'), 'layer')
        _assert_equal(f'{tag}.lambda_region', config.lambda_region, 0.0)
        _assert_equal(f'{tag}.lambda_compact', config.lambda_compact, 0.0)
        _assert_equal(f'{tag}.lambda_balance', config.lambda_balance, 0.0)

        expected_replace_mode = 'one_path' if tag == 'V4' else 'two_paths'
        _assert_equal(f'{tag}.replace_mode', sp_cfg.get('replace_mode'), expected_replace_mode)

        if config.exp_name in seen_exp_names:
            raise AssertionError(f'duplicate exp_name: {config.exp_name}')
        if config.output_dir in seen_output_dirs:
            raise AssertionError(f'duplicate output_dir: {config.output_dir}')
        seen_exp_names.add(config.exp_name)
        seen_output_dirs.add(config.output_dir)

        model = _build_model(config)
        vssm = model.vmunet
        _assert_equal(f'{tag}.bottleneck_depth', vssm.bottleneck_depth, model_cfg['depths'][-1])
        _assert_equal(
            f'{tag}.enabled_sp_scan_block_indices',
            tuple(vssm.enabled_sp_scan_block_indices),
            tuple(range(vssm.bottleneck_depth)),
        )

        enabled_modules = [
            module for module in vssm.layers[-1].modules()
            if isinstance(module, SS2D) and getattr(module, 'use_sp_scan', False)
        ]
        _assert_equal(f'{tag}.enabled_module_count', len(enabled_modules), vssm.bottleneck_depth)

        print(
            f"{tag}: exp_name={config.exp_name}, replace_mode={expected_replace_mode}, "
            f"bottleneck_depth={vssm.bottleneck_depth}, "
            f"enabled_sp_scan_block_indices={vssm.enabled_sp_scan_block_indices}"
        )

    print('VSSBlock SPScan bottleneck extension config check passed.')


if __name__ == '__main__':
    main()
