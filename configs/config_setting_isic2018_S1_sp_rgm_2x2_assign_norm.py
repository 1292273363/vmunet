from .config_setting_isic2018_R1_sp_rgm_regions_2x2_tau0p2_lite_loss import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """S1: SoftSLIC assignment distance normalization fix."""

    exp_name = 'S1_sp_rgm_2x2_assign_norm'
    output_dir = './outputs/isic2018/S1_sp_rgm_2x2_assign_norm'
    work_dir = output_dir + '/'

    use_sp_rgm = True
    return_aux = True
    num_regions = (2, 2)
    num_iters = 5
    tau = 0.2
    xy_weight = 2.0
    feat_weight = 0.2
    normalize_assign = True
    assign_norm = 'layer'
    return_distance_stats = True
    k_spatial = 3
    k_feature = 3
    path_modes = ['yx', 'xy', 'graph', 'reverse_graph']
    num_paths = len(path_modes)
    init_gamma = 1e-3
    lambda_region = 0.05
    lambda_compact = 0.0
    lambda_balance = 0.0
    test_checkpoint_type = 'best_loss'

    sp_rgm_cfg = dict(
        base_setting_config.sp_rgm_cfg,
        num_regions=num_regions,
        num_iters=num_iters,
        tau=tau,
        xy_weight=xy_weight,
        feat_weight=feat_weight,
        normalize_assign=normalize_assign,
        assign_norm=assign_norm,
        return_distance_stats=return_distance_stats,
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
