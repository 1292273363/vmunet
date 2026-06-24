from .config_setting_isic2018_F1_vssblock_spscan_stage2_last_one_path import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """F2: two-path replacement SPScan on the final encoder stage2 VSSBlock."""

    exp_name = 'F2_vssblock_spscan_stage2_last_two_paths'
    output_dir = './outputs/isic2018/F2_vssblock_spscan_stage2_last_two_paths'
    work_dir = output_dir + '/'

    sp_scan_cfg = dict(
        base_setting_config.sp_scan_cfg,
        replace_mode='two_paths',
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
