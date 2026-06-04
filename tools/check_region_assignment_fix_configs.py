import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import load_config_class


CONFIGS = {
    'R1': 'config_setting_isic2018_R1_sp_rgm_regions_2x2_tau0p2_lite_loss.py',
    'R2': 'config_setting_isic2018_R2_sp_rgm_4x4_tau0p2_balance.py',
    'R3': 'config_setting_isic2018_R3_sp_rgm_2x2_tau0p2_balance.py',
    'R4': 'config_setting_isic2018_R4_sp_rgm_4x4_tau0p2_xy4_balance.py',
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
    _expect(hasattr(config, 'lambda_region'), f'{tag}: missing lambda_region')
    _expect(hasattr(config, 'lambda_compact'), f'{tag}: missing lambda_compact')
    _expect(hasattr(config, 'lambda_balance'), f'{tag}: missing lambda_balance')
    _expect(sp_cfg.get('allow_gru_fallback') is False, f'{tag}: formal config must not allow GRU fallback')
    _expect(sp_cfg.get('path_modes') == EXPECTED_PATH_MODES, f'{tag}: path_modes mismatch')
    _expect(config.work_dir.startswith(config.output_dir), f'{tag}: work_dir should be derived from output_dir')
    return sp_cfg


def _check_specific(tag, config, sp_cfg):
    if tag == 'R1':
        _expect(tuple(sp_cfg['num_regions']) == (2, 2), 'R1: num_regions must be (2, 2)')
        _expect(_close(config.lambda_balance, 0.0), 'R1: lambda_balance must be 0.0')
    elif tag == 'R2':
        _expect(tuple(sp_cfg['num_regions']) == (4, 4), 'R2: num_regions must be (4, 4)')
        _expect(_close(config.lambda_balance, 0.0005), 'R2: lambda_balance must be 0.0005')
    elif tag == 'R3':
        _expect(tuple(sp_cfg['num_regions']) == (2, 2), 'R3: num_regions must be (2, 2)')
        _expect(_close(config.lambda_balance, 0.0005), 'R3: lambda_balance must be 0.0005')
    elif tag == 'R4':
        _expect(tuple(sp_cfg['num_regions']) == (4, 4), 'R4: num_regions must be (4, 4)')
        _expect(_close(sp_cfg['xy_weight'], 4.0), 'R4: xy_weight must be 4.0')


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
            f"k_spatial={sp_cfg['k_spatial']}, k_feature={sp_cfg['k_feature']}, "
            f"lambda_region={config.lambda_region}, lambda_compact={config.lambda_compact}, "
            f"lambda_balance={config.lambda_balance}, allow_gru_fallback={sp_cfg['allow_gru_fallback']}, "
            f"path_modes={sp_cfg['path_modes']}"
        )

    print('region assignment fix config check passed')


if __name__ == '__main__':
    main()
