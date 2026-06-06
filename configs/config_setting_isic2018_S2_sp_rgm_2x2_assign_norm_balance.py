from .config_setting_isic2018_S1_sp_rgm_2x2_assign_norm import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """S2: SoftSLIC distance fix plus lightweight mass balance."""

    exp_name = 'S2_sp_rgm_2x2_assign_norm_balance'
    output_dir = './outputs/isic2018/S2_sp_rgm_2x2_assign_norm_balance'
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
