from .config_setting_isic2018_S1b_sp_rgm_2x2_assign_norm_feat0p1 import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """S1b-NL: S1b without region reconstruction auxiliary loss."""

    exp_name = 'S1b_no_region_loss'
    output_dir = './outputs/isic2018/S1b_no_region_loss'
    work_dir = output_dir + '/'

    use_sp_rgm = True
    lambda_region = 0.0
    lambda_compact = 0.0
    lambda_balance = 0.0
    path_modes = ['yx', 'xy', 'graph', 'reverse_graph']
    num_paths = len(path_modes)
    test_checkpoint_type = 'all'

    sp_rgm_cfg = dict(
        base_setting_config.sp_rgm_cfg,
        path_modes=path_modes,
        allow_gru_fallback=False,
    )
    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=True,
        sp_rgm_cfg=sp_rgm_cfg,
    )
