import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.vmunet.vmamba import SS2D
from models.vmunet.vmunet import VMUNet
from utils import load_config_class


CONFIGS = {
    'E1': 'config_setting_isic2018_E1_vssblock_extra_graph_bottleneck_last2.py',
    'E2': 'config_setting_isic2018_E2_vssblock_extra_graph_reverse_bottleneck_last2.py',
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


def _enabled_ss2d(model):
    return [
        module for module in model.vmunet.layers[-1].modules()
        if isinstance(module, SS2D) and getattr(module, 'use_sp_scan', False)
    ]


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
        _assert_equal(f'{tag}.mode', sp_cfg.get('mode'), 'extra_path')
        _assert_equal(f'{tag}.sp_scan_stage', config.sp_scan_stage, 'bottleneck')
        _assert_equal(f'{tag}.sp_scan_blocks', config.sp_scan_blocks, 'last2')
        _assert_equal(f'{tag}.num_regions', tuple(sp_cfg.get('num_regions')), (2, 2))
        _assert_equal(f'{tag}.num_iters', sp_cfg.get('num_iters'), 5)
        _assert_equal(f'{tag}.tau', sp_cfg.get('tau'), 0.2)
        _assert_equal(f'{tag}.xy_weight', sp_cfg.get('xy_weight'), 2.0)
        _assert_equal(f'{tag}.feat_weight', sp_cfg.get('feat_weight'), 0.1)
        _assert_equal(f'{tag}.normalize_assign', sp_cfg.get('normalize_assign'), True)
        _assert_equal(f'{tag}.assign_norm', sp_cfg.get('assign_norm'), 'layer')
        _assert_equal(f'{tag}.gamma_sp_init', sp_cfg.get('gamma_sp_init'), 1e-3)
        _assert_equal(f'{tag}.lambda_region', config.lambda_region, 0.0)
        _assert_equal(f'{tag}.lambda_compact', config.lambda_compact, 0.0)
        _assert_equal(f'{tag}.lambda_balance', config.lambda_balance, 0.0)

        expected_paths = ['graph'] if tag == 'E1' else ['graph', 'reverse_graph']
        _assert_equal(f'{tag}.extra_path_types', sp_cfg.get('extra_path_types'), expected_paths)
        if config.exp_name in seen_exp_names or config.output_dir in seen_output_dirs:
            raise AssertionError(f'{tag}: exp_name or output_dir is not unique.')
        seen_exp_names.add(config.exp_name)
        seen_output_dirs.add(config.output_dir)

        model = _build_model(config)
        baseline_model = _build_model(config, use_sp_scan=False)
        vssm = model.vmunet
        _assert_equal(f'{tag}.bottleneck_depth', vssm.bottleneck_depth, model_cfg['depths'][-1])
        expected_indices = list(range(max(vssm.bottleneck_depth - 2, 0), vssm.bottleneck_depth))
        _assert_equal(
            f'{tag}.enabled_sp_scan_block_indices',
            vssm.enabled_sp_scan_block_indices,
            expected_indices,
        )

        enabled_modules = _enabled_ss2d(model)
        _assert_equal(f'{tag}.enabled_module_count', len(enabled_modules), len(expected_indices))
        for module in enabled_modules:
            _assert_equal(f'{tag}.module_mode', module.sp_scan_mode, 'extra_path')
            _assert_equal(f'{tag}.module_extra_path_types', list(module.extra_path_types), expected_paths)
            if module.gamma_graph is None or not module.gamma_graph.requires_grad:
                raise AssertionError(f'{tag}: gamma_graph is missing or frozen.')
            if tag == 'E2' and (module.gamma_reverse_graph is None or not module.gamma_reverse_graph.requires_grad):
                raise AssertionError(f'{tag}: gamma_reverse_graph is missing or frozen.')

        _assert_equal(f'{tag}.core_parameter_shapes', _core_shapes(model), _core_shapes(baseline_model))
        expected_gamma_count = len(expected_indices) * len(expected_paths)
        actual_gamma_count = sum(
            1 for name, _ in model.named_parameters()
            if name.endswith('gamma_graph') or name.endswith('gamma_reverse_graph')
        )
        _assert_equal(f'{tag}.extra_gamma_parameter_count', actual_gamma_count, expected_gamma_count)

        print(
            f"{tag}: extra_path_types={expected_paths}, bottleneck_depth={vssm.bottleneck_depth}, "
            f"enabled_sp_scan_block_indices={vssm.enabled_sp_scan_block_indices}, "
            f"extra_gamma_parameters={actual_gamma_count}"
        )

    print('VSSBlock extra-path E config check passed.')


if __name__ == '__main__':
    main()
