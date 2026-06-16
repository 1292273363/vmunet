import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet

from engine import *
import os
import sys
import argparse

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


def _extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'model' in checkpoint:
            return checkpoint['model']
    return checkpoint


def _save_metric_checkpoint(path, model, epoch, metrics, checkpoint_type):
    torch.save(
        {
            'checkpoint_type': checkpoint_type,
            'epoch': epoch,
            'loss': metrics['loss'],
            'dice': metrics.get('dice'),
            'iou': metrics.get('iou'),
            'jaccard': metrics.get('jaccard', metrics.get('iou')),
            'accuracy': metrics.get('accuracy'),
            'specificity': metrics.get('specificity'),
            'sensitivity': metrics.get('sensitivity'),
            'confusion_matrix': metrics.get('confusion_matrix'),
            'sp_scan_stats': metrics.get('sp_scan_stats'),
            'model_state_dict': model.state_dict(),
        },
        path,
    )


def _format_best_record(record):
    if record['epoch'] is None or record['loss'] is None:
        return 'not available'
    return (
        f"epoch={record['epoch']}, loss={record['loss']:.4f}, "
        f"dice={record['dice']:.4f}, iou={record['iou']:.4f}"
    )


def _get_sp_scan_stats_for_checkpoint(model, config):
    if not getattr(config, 'use_sp_scan', False):
        return None
    getter = getattr(model, 'get_sp_scan_stats', None)
    if getter is None and hasattr(model, 'module'):
        getter = getattr(model.module, 'get_sp_scan_stats', None)
    if getter is None:
        return None
    return getter()



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)





    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
            use_sp_rgm=model_cfg.get('use_sp_rgm', False),
            sp_rgm_cfg=model_cfg.get('sp_rgm_cfg'),
            use_sp_scan=model_cfg.get('use_sp_scan', getattr(config, 'use_sp_scan', False)),
            sp_scan_cfg=model_cfg.get('sp_scan_cfg', getattr(config, 'sp_scan_cfg', None)),
            sp_scan_stage=model_cfg.get('sp_scan_stage', getattr(config, 'sp_scan_stage', None)),
            sp_scan_blocks=model_cfg.get('sp_scan_blocks', getattr(config, 'sp_scan_blocks', None)),
        )
        model.load_from()
        
    else: raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 256, logger)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    start_epoch = 1
    best_records = {
        'best_loss': {'value': float('inf'), 'epoch': None, 'loss': None, 'dice': None, 'iou': None},
        'best_dice': {'value': -1.0, 'epoch': None, 'loss': None, 'dice': None, 'iou': None},
        'best_iou': {'value': -1.0, 'epoch': None, 'loss': None, 'dice': None, 'iou': None},
    }

    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(_extract_model_state(checkpoint))
        config.work_dir = config.img_save_path
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        return




    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        if 'best_records' in checkpoint:
            best_records = checkpoint['best_records']
        else:
            best_records['best_loss'].update({
                'value': checkpoint.get('min_loss', float('inf')),
                'epoch': checkpoint.get('min_epoch'),
                'loss': checkpoint.get('loss', 0.0),
                'dice': checkpoint.get('dice', 0.0),
                'iou': checkpoint.get('iou', 0.0),
            })
        loss = checkpoint.get('loss', 0.0)

        log_info = (
            f"resuming model from {resume_model}. resume_epoch: {saved_epoch}, "
            f"best_loss: {_format_best_record(best_records['best_loss'])}, loss: {loss:.4f}"
        )
        logger.info(log_info)




    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        val_metrics = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )
        sp_scan_stats = _get_sp_scan_stats_for_checkpoint(model, config)
        if sp_scan_stats is not None:
            val_metrics['sp_scan_stats'] = sp_scan_stats

        if val_metrics['loss'] < best_records['best_loss']['value']:
            best_records['best_loss'] = {
                'value': val_metrics['loss'],
                'epoch': epoch,
                'loss': val_metrics['loss'],
                'dice': val_metrics['dice'],
                'iou': val_metrics['iou'],
            }
            _save_metric_checkpoint(
                os.path.join(checkpoint_dir, 'best_loss.pth'),
                model,
                epoch,
                val_metrics,
                'best_loss',
            )
            logger.info(f"saved best_loss.pth: {_format_best_record(best_records['best_loss'])}")

        if val_metrics['dice'] > best_records['best_dice']['value']:
            best_records['best_dice'] = {
                'value': val_metrics['dice'],
                'epoch': epoch,
                'loss': val_metrics['loss'],
                'dice': val_metrics['dice'],
                'iou': val_metrics['iou'],
            }
            _save_metric_checkpoint(
                os.path.join(checkpoint_dir, 'best_dice.pth'),
                model,
                epoch,
                val_metrics,
                'best_dice',
            )
            logger.info(f"saved best_dice.pth: {_format_best_record(best_records['best_dice'])}")

        if val_metrics['iou'] > best_records['best_iou']['value']:
            best_records['best_iou'] = {
                'value': val_metrics['iou'],
                'epoch': epoch,
                'loss': val_metrics['loss'],
                'dice': val_metrics['dice'],
                'iou': val_metrics['iou'],
            }
            _save_metric_checkpoint(
                os.path.join(checkpoint_dir, 'best_iou.pth'),
                model,
                epoch,
                val_metrics,
                'best_iou',
            )
            logger.info(f"saved best_iou.pth: {_format_best_record(best_records['best_iou'])}")

        last_checkpoint = {
            'epoch': epoch,
            'min_loss': best_records['best_loss']['value'],
            'min_epoch': best_records['best_loss']['epoch'],
            'loss': val_metrics['loss'],
            'dice': val_metrics['dice'],
            'iou': val_metrics['iou'],
            'best_records': best_records,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(last_checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))
        torch.save(last_checkpoint, os.path.join(checkpoint_dir, 'last.pth')) 

    for checkpoint_type in ('best_loss', 'best_dice', 'best_iou'):
        checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint_type}.pth')
        log_info = (
            f"{checkpoint_type} checkpoint: {checkpoint_path}, "
            f"{_format_best_record(best_records[checkpoint_type])}"
        )
        print(log_info)
        logger.info(log_info)

    selected_checkpoint_type = getattr(config, 'test_checkpoint_type', 'best_loss')
    valid_checkpoint_types = {'best_loss', 'best_dice', 'best_iou', 'all'}
    if selected_checkpoint_type not in valid_checkpoint_types:
        raise ValueError(
            "config.test_checkpoint_type must be one of "
            "['best_loss', 'best_dice', 'best_iou', 'all']"
        )

    test_checkpoint_types = (
        ('best_loss', 'best_dice', 'best_iou')
        if selected_checkpoint_type == 'all'
        else (selected_checkpoint_type,)
    )
    for checkpoint_type in test_checkpoint_types:
        selected_checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint_type}.pth')
        if not os.path.exists(selected_checkpoint_path):
            logger.info(f'skipped final test because {selected_checkpoint_path} does not exist.')
            continue

        print('#----------Testing----------#')
        best_weight = torch.load(selected_checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(_extract_model_state(best_weight))
        config.active_test_checkpoint_type = checkpoint_type
        test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VM-UNet on ISIC datasets.')
    parser.add_argument(
        '--config',
        default='config_setting',
        help='Config module or config filename under configs/. Example: config_setting_isic2018_A4_sp_rgm_multi_path_4x4.py',
    )
    parser.add_argument(
        '--test-checkpoint-type',
        choices=['best_loss', 'best_dice', 'best_iou', 'all'],
        default=None,
        help='Checkpoint type used for the final test after training.',
    )
    args = parser.parse_args()
    config = load_config_class(args.config, default_module='configs.config_setting')
    if args.test_checkpoint_type is not None:
        config.test_checkpoint_type = args.test_checkpoint_type
    if getattr(config, 'supported_by_current_core', True) is False:
        raise RuntimeError(f"{config.exp_name}: {config.unsupported_reason}")
    main(config)
