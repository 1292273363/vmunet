from .config_setting_isic2018_A4_sp_rgm_multi_path_4x4 import setting_config as base_setting_config


class setting_config(base_setting_config):
    """C2: Larger residual gamma for faster SP-RGM injection."""

    exp_name = 'C2_sp_rgm_gamma_0p01_4x4'
    output_dir = './outputs/isic2018/C2_sp_rgm_gamma_0p01_4x4'
    work_dir = output_dir + '/'

    init_gamma = 0.01
    sp_rgm_cfg = dict(base_setting_config.sp_rgm_cfg, init_gamma=init_gamma)
    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=base_setting_config.use_sp_rgm,
        sp_rgm_cfg=sp_rgm_cfg,
    )
