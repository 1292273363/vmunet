import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.vmunet.vmamba import SS2D
from models.vmunet.vmunet import VMUNet
from utils import load_config_class


CONFIGS = {
    'F1': 'config_setting_isic2018_F1_vssblock_spscan_stage2_last_one_path.py',
    'F2': 'config_setting_isic2018_F2_vssblock_spscan_stage2_last_two_paths.py',
}
CORE_PARAM_NAMES = {
    'x_proj_weight',
    'dt_projs_weight',
    'dt_projs_bias',
    'A_logs',
    'Ds',
}


def _assert_equal(name, actual, expected):
    if actual != expected:
        raise AssertionError(f'{name}: expected {expected!r}, got {actual!r}')


def _build_model(config, use_sp_scan=None):
    model_cfg = config.model_config
    if use_sp_scan is None:
        use_sp_scan = model_cfg.get('use_sp_scan', False)
    return VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=None,
        use_sp_rgm=False,
        sp_rgm_cfg=None,
        use_sp_scan=use_sp_scan,
        sp_scan_cfg=model_cfg.get('sp_scan_cfg') if use_sp_scan else None,
        sp_scan_stage=model_cfg.get('sp_scan_stage') if use_sp_scan else None,
        sp_scan_blocks=model_cfg.get('sp_scan_blocks') if use_sp_scan else None,
    )


def _enabled_ss2d_indices(model):
    enabled = []
    for stage_idx, layer in enumerate(model.vmunet.layers):
        for block_idx, block in enumerate(layer.blocks):
            if isinstance(block.self_attention, SS2D) and block.self_attention.use_sp_scan:
                enabled.append((stage_idx, block_idx))
    return enabled


def _core_shapes(model):
    return {
        name: tuple(param.shape)
        for name, param in model.named_parameters()
        if name.split('.')[-1] in CORE_PARAM_NAMES
    }


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
        _assert_equal(f'{tag}.sp_scan_stage', config.sp_scan_stage, 'stage2')
        _assert_equal(f'{tag}.sp_scan_blocks', config.sp_scan_blocks, 'last')
        _assert_equal(f'{tag}.mode', sp_cfg.get('mode'), 'replacement')
        _assert_equal(f'{tag}.num_regions', tuple(sp_cfg.get('num_regions')), (4, 4))
        _assert_equal(f'{tag}.num_iters', sp_cfg.get('num_iters'), 5)
        _assert_equal(f'{tag}.tau', sp_cfg.get('tau'), 0.2)
        _assert_equal(f'{tag}.xy_weight', sp_cfg.get('xy_weight'), 2.0)
        _assert_equal(f'{tag}.feat_weight', sp_cfg.get('feat_weight'), 0.1)
        _assert_equal(f'{tag}.normalize_assign', sp_cfg.get('normalize_assign'), True)
        _assert_equal(f'{tag}.assign_norm', sp_cfg.get('assign_norm'), 'layer')
        _assert_equal(f'{tag}.lambda_region', config.lambda_region, 0.0)
        _assert_equal(f'{tag}.lambda_compact', config.lambda_compact, 0.0)
        _assert_equal(f'{tag}.lambda_balance', config.lambda_balance, 0.0)

        expected_replace_mode = 'one_path' if tag == 'F1' else 'two_paths'
        _assert_equal(f'{tag}.replace_mode', sp_cfg.get('replace_mode'), expected_replace_mode)
        if config.exp_name in seen_exp_names or config.output_dir in seen_output_dirs:
            raise AssertionError(f'{tag}: exp_name or output_dir is not unique.')
        seen_exp_names.add(config.exp_name)
        seen_output_dirs.add(config.output_dir)

        model = _build_model(config)
        baseline_model = _build_model(config, use_sp_scan=False)
        vssm = model.vmunet
        _assert_equal(f'{tag}.enabled_stage_index', vssm.enabled_sp_scan_stage_index, 2)
        _assert_equal(f'{tag}.stage2_depth', vssm.stage2_depth, model_cfg['depths'][2])
        _assert_equal(f'{tag}.enabled_block_indices', vssm.enabled_sp_scan_block_indices, [vssm.stage2_depth - 1])
        _assert_equal(f'{tag}.enabled_ss2d', _enabled_ss2d_indices(model), [(2, vssm.stage2_depth - 1)])
        _assert_equal(f'{tag}.baseline_enabled_ss2d', _enabled_ss2d_indices(baseline_model), [])
        _assert_equal(f'{tag}.core_parameter_shapes', _core_shapes(model), _core_shapes(baseline_model))

        print(
            f"{tag}: replace_mode={expected_replace_mode}, stage2_depth={vssm.stage2_depth}, "
            f"enabled_sp_scan_stage_index={vssm.enabled_sp_scan_stage_index}, "
            f"enabled_sp_scan_block_indices={vssm.enabled_sp_scan_block_indices}"
        )

    print('VSSBlock stage2 F config check passed.')


if __name__ == '__main__':
    main()
