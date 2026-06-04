from .config_setting_isic2018_R2_sp_rgm_4x4_tau0p2_balance import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """R4: optional stronger spatial weighting for 4x4 region assignment."""

    exp_name = 'R4_sp_rgm_4x4_tau0p2_xy4_balance'
    output_dir = './outputs/isic2018/R4_sp_rgm_4x4_tau0p2_xy4_balance'
    work_dir = output_dir + '/'

    xy_weight = 4.0

    sp_rgm_cfg = dict(
        base_setting_config.sp_rgm_cfg,
        xy_weight=xy_weight,
        allow_gru_fallback=False,
    )
    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=True,
        sp_rgm_cfg=sp_rgm_cfg,
    )
