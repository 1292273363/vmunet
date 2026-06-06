from .config_setting_isic2018_S1_sp_rgm_2x2_assign_norm import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """S1b: optional lower feature weight if S1 assignment remains too hard."""

    exp_name = 'S1b_sp_rgm_2x2_assign_norm_feat0p1'
    output_dir = './outputs/isic2018/S1b_sp_rgm_2x2_assign_norm_feat0p1'
    work_dir = output_dir + '/'

    feat_weight = 0.1

    sp_rgm_cfg = dict(
        base_setting_config.sp_rgm_cfg,
        feat_weight=feat_weight,
        allow_gru_fallback=False,
    )
    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=True,
        sp_rgm_cfg=sp_rgm_cfg,
    )
