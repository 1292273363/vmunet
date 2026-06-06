import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import load_config_class


CONFIGS = {
    'S1': 'config_setting_isic2018_S1_sp_rgm_2x2_assign_norm.py',
    'S2': 'config_setting_isic2018_S2_sp_rgm_2x2_assign_norm_balance.py',
    'S1b': 'config_setting_isic2018_S1b_sp_rgm_2x2_assign_norm_feat0p1.py',
}
EXPECTED_PATH_MODES = ['yx', 'xy', 'graph', 'reverse_graph']


def _expect(condition, message):
    if not condition:
        raise AssertionError(message)


def _close(actual, expected, tol=1e-12):
    return abs(float(actual) - float(expected)) <= tol


def _check_common(tag, config):
    model_cfg = config.model_config
    _expect(config.use_sp_rgm is True, f'{tag}: use_sp_rgm must be True')
    _expect(model_cfg.get('use_sp_rgm') is True, f'{tag}: model_config["use_sp_rgm"] must be True')
    _expect('sp_rgm_cfg' in model_cfg, f'{tag}: missing model_config["sp_rgm_cfg"]')

    sp_cfg = model_cfg['sp_rgm_cfg']
    _expect(sp_cfg is not None, f'{tag}: sp_rgm_cfg is None')
    _expect('num_iters' in sp_cfg, f'{tag}: sp_rgm_cfg must use num_iters')
    _expect('slic_iters' not in sp_cfg, f'{tag}: sp_rgm_cfg must not use slic_iters')
    _expect(tuple(sp_cfg['num_regions']) == (2, 2), f'{tag}: num_regions must be (2, 2)')
    _expect(sp_cfg['num_iters'] == 5, f'{tag}: num_iters must be 5')
    _expect(_close(sp_cfg['tau'], 0.2), f'{tag}: tau must be 0.2')
    _expect(_close(sp_cfg['xy_weight'], 2.0), f'{tag}: xy_weight must be 2.0')
    _expect('feat_weight' in sp_cfg, f'{tag}: missing feat_weight')
    _expect(sp_cfg.get('normalize_assign') is True, f'{tag}: normalize_assign must be True')
    _expect(sp_cfg.get('assign_norm') == 'layer', f'{tag}: assign_norm must be "layer"')
    _expect(sp_cfg.get('return_distance_stats') is True, f'{tag}: return_distance_stats must be True')
    _expect(sp_cfg.get('allow_gru_fallback') is False, f'{tag}: formal config must not allow GRU fallback')
    _expect(sp_cfg.get('path_modes') == EXPECTED_PATH_MODES, f'{tag}: path_modes mismatch')
    _expect(hasattr(config, 'lambda_region'), f'{tag}: missing lambda_region')
    _expect(hasattr(config, 'lambda_compact'), f'{tag}: missing lambda_compact')
    _expect(hasattr(config, 'lambda_balance'), f'{tag}: missing lambda_balance')
    _expect(config.work_dir.startswith(config.output_dir), f'{tag}: work_dir should be derived from output_dir')
    return sp_cfg


def _check_specific(tag, config, sp_cfg):
    if tag == 'S1':
        _expect(_close(sp_cfg['feat_weight'], 0.2), 'S1: feat_weight must be 0.2')
        _expect(_close(config.lambda_balance, 0.0), 'S1: lambda_balance must be 0.0')
    elif tag == 'S2':
        _expect(_close(sp_cfg['feat_weight'], 0.2), 'S2: feat_weight must be 0.2')
        _expect(_close(config.lambda_balance, 0.0005), 'S2: lambda_balance must be 0.0005')
    elif tag == 'S1b':
        _expect(_close(sp_cfg['feat_weight'], 0.1), 'S1b: feat_weight must be 0.1')
        _expect(_close(config.lambda_balance, 0.0), 'S1b: lambda_balance must be 0.0')


def main():
    seen_dirs = {}
    for tag, config_name in CONFIGS.items():
        config = load_config_class(config_name)
        sp_cfg = _check_common(tag, config)
        _check_specific(tag, config, sp_cfg)
        duplicate_tag = seen_dirs.get(config.output_dir)
        _expect(duplicate_tag is None, f'{tag}: duplicate output_dir with {duplicate_tag}')
        seen_dirs[config.output_dir] = tag

        print(
            f"{tag}: exp_name={config.exp_name}, output_dir={config.output_dir}, "
            f"num_regions={tuple(sp_cfg['num_regions'])}, num_iters={sp_cfg['num_iters']}, "
            f"tau={sp_cfg['tau']}, xy_weight={sp_cfg['xy_weight']}, "
            f"feat_weight={sp_cfg['feat_weight']}, normalize_assign={sp_cfg['normalize_assign']}, "
            f"assign_norm={sp_cfg['assign_norm']}, return_distance_stats={sp_cfg['return_distance_stats']}, "
            f"k_spatial={sp_cfg['k_spatial']}, k_feature={sp_cfg['k_feature']}, "
            f"lambda_region={config.lambda_region}, lambda_compact={config.lambda_compact}, "
            f"lambda_balance={config.lambda_balance}, allow_gru_fallback={sp_cfg['allow_gru_fallback']}, "
            f"path_modes={sp_cfg['path_modes']}"
        )

    print('softslic distance fix config check passed')


if __name__ == '__main__':
    main()
