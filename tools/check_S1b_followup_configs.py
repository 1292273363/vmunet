import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import load_config_class


CONFIGS = {
    'S1b-NL': 'config_setting_isic2018_S1b_no_region_loss.py',
    'S2b': 'config_setting_isic2018_S2b_assign_norm_balance.py',
    'S1b-G': 'config_setting_isic2018_S1b_graph_path_only.py',
    'S1b-YXG': 'config_setting_isic2018_S1b_yx_graph_paths.py',
}


def _expect(condition, message):
    if not condition:
        raise AssertionError(message)


def _close(actual, expected, tol=1e-12):
    return abs(float(actual) - float(expected)) <= tol


def _check_common(tag, config):
    model_cfg = config.model_config
    _expect(config.use_sp_rgm is True, f'{tag}: use_sp_rgm must be True')
    _expect(model_cfg.get('use_sp_rgm') is True, f'{tag}: model_config["use_sp_rgm"] must be True')
    _expect('sp_rgm_cfg' in model_cfg, f'{tag}: missing sp_rgm_cfg')

    sp_cfg = model_cfg['sp_rgm_cfg']
    _expect(sp_cfg is not None, f'{tag}: sp_rgm_cfg is None')
    _expect(tuple(sp_cfg['num_regions']) == (2, 2), f'{tag}: num_regions must be (2, 2)')
    _expect(sp_cfg['num_iters'] == 5, f'{tag}: num_iters must be 5')
    _expect('slic_iters' not in sp_cfg, f'{tag}: use num_iters, not slic_iters')
    _expect(_close(sp_cfg['tau'], 0.2), f'{tag}: tau must be 0.2')
    _expect(_close(sp_cfg['xy_weight'], 2.0), f'{tag}: xy_weight must be 2.0')
    _expect(_close(sp_cfg['feat_weight'], 0.1), f'{tag}: feat_weight must be 0.1')
    _expect(sp_cfg.get('normalize_assign') is True, f'{tag}: normalize_assign must be True')
    _expect(sp_cfg.get('assign_norm') == 'layer', f'{tag}: assign_norm must be "layer"')
    _expect(sp_cfg.get('return_distance_stats') is True, f'{tag}: return_distance_stats must be True')
    _expect(sp_cfg.get('allow_gru_fallback') is False, f'{tag}: formal config must not allow GRU fallback')
    _expect(_close(config.lambda_compact, 0.0), f'{tag}: lambda_compact must be 0.0')
    _expect(getattr(config, 'test_checkpoint_type', None) == 'all', f'{tag}: test_checkpoint_type must be "all"')
    return sp_cfg


def _check_specific(tag, config, sp_cfg):
    if tag == 'S1b-NL':
        _expect(_close(config.lambda_region, 0.0), 'S1b-NL: lambda_region must be 0.0')
        _expect(_close(config.lambda_balance, 0.0), 'S1b-NL: lambda_balance must be 0.0')
        _expect(sp_cfg['path_modes'] == ['yx', 'xy', 'graph', 'reverse_graph'], 'S1b-NL: path_modes mismatch')
    elif tag == 'S2b':
        _expect(_close(config.lambda_region, 0.05), 'S2b: lambda_region must be 0.05')
        _expect(_close(config.lambda_balance, 0.0005), 'S2b: lambda_balance must be 0.0005')
        _expect(sp_cfg['path_modes'] == ['yx', 'xy', 'graph', 'reverse_graph'], 'S2b: path_modes mismatch')
    elif tag == 'S1b-G':
        _expect(_close(config.lambda_region, 0.05), 'S1b-G: lambda_region must be 0.05')
        _expect(_close(config.lambda_balance, 0.0), 'S1b-G: lambda_balance must be 0.0')
        _expect(sp_cfg['path_modes'] == ['graph'], 'S1b-G: path_modes must be ["graph"]')
    elif tag == 'S1b-YXG':
        _expect(_close(config.lambda_region, 0.05), 'S1b-YXG: lambda_region must be 0.05')
        _expect(_close(config.lambda_balance, 0.0), 'S1b-YXG: lambda_balance must be 0.0')
        _expect(sp_cfg['path_modes'] == ['yx', 'graph'], 'S1b-YXG: path_modes must be ["yx", "graph"]')


def main():
    seen_output_dirs = {}
    seen_exp_names = {}
    for tag, config_name in CONFIGS.items():
        config = load_config_class(config_name)
        sp_cfg = _check_common(tag, config)
        _check_specific(tag, config, sp_cfg)

        duplicate_dir = seen_output_dirs.get(config.output_dir)
        _expect(duplicate_dir is None, f'{tag}: duplicate output_dir with {duplicate_dir}')
        seen_output_dirs[config.output_dir] = tag

        duplicate_name = seen_exp_names.get(config.exp_name)
        _expect(duplicate_name is None, f'{tag}: duplicate exp_name with {duplicate_name}')
        seen_exp_names[config.exp_name] = tag

        print(
            f"{tag}: exp_name={config.exp_name}, output_dir={config.output_dir}, "
            f"lambda_region={config.lambda_region}, lambda_compact={config.lambda_compact}, "
            f"lambda_balance={config.lambda_balance}, test_checkpoint_type={config.test_checkpoint_type}, "
            f"num_regions={tuple(sp_cfg['num_regions'])}, num_iters={sp_cfg['num_iters']}, "
            f"tau={sp_cfg['tau']}, xy_weight={sp_cfg['xy_weight']}, feat_weight={sp_cfg['feat_weight']}, "
            f"normalize_assign={sp_cfg['normalize_assign']}, assign_norm={sp_cfg['assign_norm']}, "
            f"return_distance_stats={sp_cfg['return_distance_stats']}, "
            f"path_modes={sp_cfg['path_modes']}, allow_gru_fallback={sp_cfg['allow_gru_fallback']}"
        )

    print('S1b follow-up config check passed')


if __name__ == '__main__':
    main()
