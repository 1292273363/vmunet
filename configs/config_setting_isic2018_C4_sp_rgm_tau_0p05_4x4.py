from .config_setting_isic2018_A4_sp_rgm_multi_path_4x4 import setting_config as base_setting_config


class setting_config(base_setting_config):
    """C4: Harder assignments with tau=0.05."""

    exp_name = 'C4_sp_rgm_tau_0p05_4x4'
    output_dir = './outputs/isic2018/C4_sp_rgm_tau_0p05_4x4'
    work_dir = output_dir + '/'

    tau = 0.05
    sp_rgm_cfg = dict(base_setting_config.sp_rgm_cfg, tau=tau)
    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=base_setting_config.use_sp_rgm,
        sp_rgm_cfg=sp_rgm_cfg,
    )
