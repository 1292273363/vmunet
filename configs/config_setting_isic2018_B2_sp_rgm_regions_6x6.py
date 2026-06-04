from .config_setting_isic2018_A4_sp_rgm_multi_path_4x4 import setting_config as base_setting_config


class setting_config(base_setting_config):
    """B2: Region-count ablation with a 6x6 grid."""

    exp_name = 'B2_sp_rgm_regions_6x6'
    output_dir = './outputs/isic2018/B2_sp_rgm_regions_6x6'
    work_dir = output_dir + '/'

    num_regions = [6, 6]
    k_spatial = 6
    k_feature = 6
    sp_rgm_cfg = dict(
        base_setting_config.sp_rgm_cfg,
        num_regions=tuple(num_regions),
        k_spatial=k_spatial,
        k_feature=k_feature,
    )
    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=base_setting_config.use_sp_rgm,
        sp_rgm_cfg=sp_rgm_cfg,
    )
