from .config_setting_synapse import setting_config as base_setting_config


class setting_config(base_setting_config):
    """Synapse SP-RGM experiment with the recommended 4x4 region grid."""

    use_sp_rgm = True
    sp_rgm_cfg = dict(base_setting_config.sp_rgm_cfg, num_regions=(4, 4))
    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=use_sp_rgm,
        sp_rgm_cfg=sp_rgm_cfg,
    )
