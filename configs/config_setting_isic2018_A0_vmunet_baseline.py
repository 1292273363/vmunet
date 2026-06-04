from .config_setting import setting_config as base_setting_config


class setting_config(base_setting_config):
    """A0: ISIC2018 VM-UNet baseline."""

    exp_name = 'A0_vmunet_baseline'
    output_dir = './outputs/isic2018/A0_vmunet_baseline'
    work_dir = output_dir + '/'

    use_sp_rgm = False
    return_aux = False
    lambda_region = 0.0
    lambda_compact = 0.0
    test_checkpoint_type = 'best_loss'

    model_config = dict(base_setting_config.model_config, use_sp_rgm=use_sp_rgm)
