from .config_setting_isic2018_V1_vssblock_spscan_one_path import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """E1: retain four SS2D paths and add one gated graph residual path."""

    exp_name = 'E1_vssblock_extra_graph_bottleneck_last2'
    output_dir = './outputs/isic2018/E1_vssblock_extra_graph_bottleneck_last2'
    work_dir = output_dir + '/'

    use_sp_rgm = False
    use_sp_scan = True
    sp_scan_stage = 'bottleneck'
    sp_scan_blocks = 'last2'

    lambda_region = 0.0
    lambda_compact = 0.0
    lambda_balance = 0.0
    test_checkpoint_type = 'all'

    sp_scan_cfg = {
        'enabled': True,
        'mode': 'extra_path',
        'extra_path_types': ['graph'],
        'num_regions': (2, 2),
        'num_iters': 5,
        'tau': 0.2,
        'xy_weight': 2.0,
        'feat_weight': 0.1,
        'normalize_assign': True,
        'assign_norm': 'layer',
        'k_spatial': 3,
        'k_feature': 3,
        'graph_order': 'greedy',
        'token_inner_order': 'raster',
        'detach_order': True,
        'debug_stats': True,
        'gamma_sp_init': 1e-3,
    }

    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=False,
        sp_rgm_cfg=None,
        use_sp_scan=True,
        sp_scan_cfg=sp_scan_cfg,
        sp_scan_stage=sp_scan_stage,
        sp_scan_blocks=sp_scan_blocks,
    )
