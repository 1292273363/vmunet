from .config_setting_isic2018_A4_sp_rgm_multi_path_4x4 import setting_config as base_setting_config


class setting_config(base_setting_config):
    """B1: Region-count ablation with a 4x4 grid."""

    exp_name = 'B1_sp_rgm_regions_4x4'
    output_dir = './outputs/isic2018/B1_sp_rgm_regions_4x4'
    work_dir = output_dir + '/'
