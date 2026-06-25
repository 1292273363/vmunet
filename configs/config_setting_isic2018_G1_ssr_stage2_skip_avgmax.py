from .config_setting_isic2018_A0_vmunet_baseline import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """G1: SuiT-style avg+max Superpixel Skip Refinement on stage2 skip."""

    exp_name = 'G1_ssr_stage2_skip_avgmax'
    output_dir = './outputs/isic2018/G1_ssr_stage2_skip_avgmax'
    work_dir = output_dir + '/'

    use_sp_rgm = False
    use_sp_scan = False
    use_ssr = True

    lambda_region = 0.0
    lambda_compact = 0.0
    lambda_balance = 0.0
    test_checkpoint_type = 'all'

    ssr_cfg = {
        'enabled': True,
        'ssr_stages': ['stage2'],
        'num_regions': {
            'stage2': (4, 4),
        },
        'num_iters': 5,
        'tau': 0.2,
        'xy_weight': 2.0,
        'feat_weight': 0.1,
        'normalize_assign': True,
        'assign_norm': 'layer',
        'use_pos_embed': True,
        'use_avg_pool': True,
        'use_max_pool': True,
        'use_graph': False,
        'region_update': 'mlp',
        'gamma_init': 1e-3,
        'gate_type': 'bounded_tanh',
        'gate_scale': 0.1,
        'norm_type': 'group',
        'num_groups': 8,
        'detach_assignment': False,
        'debug_stats': True,
    }

    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=False,
        sp_rgm_cfg=None,
        use_sp_scan=False,
        sp_scan_cfg=None,
        sp_scan_stage=None,
        sp_scan_blocks=None,
        use_ssr=True,
        ssr_cfg=ssr_cfg,
    )
