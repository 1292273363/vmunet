from .config_setting_isic2018_R1_sp_rgm_regions_2x2_tau0p2_lite_loss import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """R3: combine 2x2 regions with lightweight mass balance loss."""

    exp_name = 'R3_sp_rgm_2x2_tau0p2_balance'
    output_dir = './outputs/isic2018/R3_sp_rgm_2x2_tau0p2_balance'
    work_dir = output_dir + '/'

    lambda_balance = 0.0005

    sp_rgm_cfg = dict(
        base_setting_config.sp_rgm_cfg,
        allow_gru_fallback=False,
    )
    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=True,
        sp_rgm_cfg=sp_rgm_cfg,
    )
