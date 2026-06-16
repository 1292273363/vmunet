from .config_setting_isic2018_A0_vmunet_baseline import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """V1: replace one SS2D path with a superpixel graph dense-token path."""

    exp_name = 'V1_vssblock_spscan_one_path'
    output_dir = './outputs/isic2018/V1_vssblock_spscan_one_path'
    work_dir = output_dir + '/'

    use_sp_rgm = False
    use_sp_scan = True
    sp_scan_stage = 'bottleneck_last'
    sp_scan_blocks = 'last'
    return_aux = False

    lambda_region = 0.0
    lambda_compact = 0.0
    lambda_balance = 0.0
    test_checkpoint_type = 'all'

    sp_scan_cfg = {
        'enabled': True,
        'replace_mode': 'one_path',
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
