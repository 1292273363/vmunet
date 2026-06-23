from .config_setting_isic2018_E1_vssblock_extra_graph_bottleneck_last2 import (
    setting_config as base_setting_config,
)


class setting_config(base_setting_config):
    """E2: retain four SS2D paths and add gated graph/reverse-graph residual paths."""

    exp_name = 'E2_vssblock_extra_graph_reverse_bottleneck_last2'
    output_dir = './outputs/isic2018/E2_vssblock_extra_graph_reverse_bottleneck_last2'
    work_dir = output_dir + '/'

    sp_scan_cfg = dict(
        base_setting_config.sp_scan_cfg,
        extra_path_types=['graph', 'reverse_graph'],
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
