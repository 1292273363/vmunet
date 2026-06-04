from .config_setting import setting_config as base_setting_config


class setting_config(base_setting_config):
    """A1: SP-RGM module only, without auxiliary region losses."""

    exp_name = 'A1_sp_rgm_module_only_4x4'
    output_dir = './outputs/isic2018/A1_sp_rgm_module_only_4x4'
    work_dir = output_dir + '/'

    use_sp_rgm = True
    return_aux = True
    num_regions = (4, 4)
    num_iters = 5
    tau = 0.1
    xy_weight = 2.0
    k_spatial = 4
    k_feature = 4
    path_modes = ['yx', 'xy', 'graph', 'reverse_graph']
    num_paths = len(path_modes)
    init_gamma = 0.001
    lambda_region = 0.0
    lambda_compact = 0.0
    test_checkpoint_type = 'best_loss'

    sp_rgm_cfg = dict(
        base_setting_config.sp_rgm_cfg,
        num_regions=num_regions,
        num_iters=num_iters,
        tau=tau,
        xy_weight=xy_weight,
        k_spatial=k_spatial,
        k_feature=k_feature,
        init_gamma=init_gamma,
        path_modes=path_modes,
        allow_gru_fallback=False,
    )
    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=use_sp_rgm,
        sp_rgm_cfg=sp_rgm_cfg,
    )
