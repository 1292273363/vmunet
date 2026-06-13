from .config_setting_isic2018_V1_vssblock_spscan_one_path import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """V2: replace two SS2D paths with graph and reverse-graph dense-token paths."""

    exp_name = 'V2_vssblock_spscan_two_paths'
    output_dir = './outputs/isic2018/V2_vssblock_spscan_two_paths'
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
    )
