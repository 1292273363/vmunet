from .config_setting_isic2018_V2_vssblock_spscan_two_paths import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """V3: two-path SPScan with spatial-only SoftSLIC assignment distance."""

    exp_name = 'V3_vssblock_spscan_two_paths_spatial'
    output_dir = './outputs/isic2018/V3_vssblock_spscan_two_paths_spatial'
    work_dir = output_dir + '/'

    sp_scan_cfg = dict(
        base_setting_config.sp_scan_cfg,
        feat_weight=0.0,
    )

    model_config = dict(
        base_setting_config.model_config,
        use_sp_rgm=False,
        sp_rgm_cfg=None,
        use_sp_scan=True,
        sp_scan_cfg=sp_scan_cfg,
        sp_scan_stage=base_setting_config.sp_scan_stage,
        sp_scan_blocks=base_setting_config.sp_scan_blocks,
    )
