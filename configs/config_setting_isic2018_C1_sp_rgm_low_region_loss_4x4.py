from .config_setting_isic2018_A4_sp_rgm_multi_path_4x4 import setting_config as base_setting_config


class setting_config(base_setting_config):
    """C1: Lower auxiliary region-loss weights for stability probing."""

    exp_name = 'C1_sp_rgm_low_region_loss_4x4'
    output_dir = './outputs/isic2018/C1_sp_rgm_low_region_loss_4x4'
    work_dir = output_dir + '/'

    lambda_region = 0.05
    lambda_compact = 0.0005
