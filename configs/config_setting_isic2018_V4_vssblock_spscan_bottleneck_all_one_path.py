from .config_setting_isic2018_V1_vssblock_spscan_one_path import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """V4: enable one-path SPScan on all VSSBlocks in the bottleneck stage."""

    exp_name = 'V4_vssblock_spscan_bottleneck_all_one_path'
    output_dir = './outputs/isic2018/V4_vssblock_spscan_bottleneck_all_one_path'
    work_dir = output_dir + '/'

    use_sp_rgm = False
    use_sp_scan = True
    sp_scan_stage = 'bottleneck'
    sp_scan_blocks = 'all'

    sp_scan_cfg = dict(
        base_setting_config.sp_scan_cfg,
        replace_mode='one_path',
    )

    lambda_region = 0.0
    lambda_compact = 0.0
    lambda_balance = 0.0
    test_checkpoint_type = 'all'

    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=False,
        sp_rgm_cfg=None,
        use_sp_scan=True,
        sp_scan_cfg=sp_scan_cfg,
        sp_scan_stage=sp_scan_stage,
        sp_scan_blocks=sp_scan_blocks,
    )
