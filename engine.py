import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
from losses.region_losses import (
    region_compactness_loss,
    region_label_reconstruction_loss,
    region_mass_balance_loss,
)


def _binary_metrics_from_arrays(preds, gts, threshold):
    """Compute binary segmentation metrics from flattened prediction arrays."""
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds >= threshold, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    dice = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    iou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    return {
        'dice': dice,
        'f1_or_dsc': dice,
        'iou': iou,
        'jaccard': iou,
        'miou': iou,
        'accuracy': accuracy,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'confusion_matrix': confusion,
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
        'TP': int(TP),
    }


def _format_eval_log(prefix, metrics):
    return (
        f"{prefix}, loss: {metrics['loss']:.4f}, "
        f"miou: {metrics['iou']}, iou_jaccard: {metrics['jaccard']}, "
        f"f1_or_dsc: {metrics['dice']}, dice: {metrics['dice']}, "
        f"accuracy: {metrics['accuracy']}, specificity: {metrics['specificity']}, "
        f"sensitivity: {metrics['sensitivity']}, "
        f"TP: {metrics['TP']}, FP: {metrics['FP']}, "
        f"TN: {metrics['TN']}, FN: {metrics['FN']}, "
        f"confusion_matrix: {metrics['confusion_matrix']}"
    )


def _format_log_value(value, precision=4):
    if value is None:
        return 'NA'
    if torch.is_tensor(value):
        if value.numel() != 1:
            return str(value)
        value = value.item()
    if isinstance(value, float):
        return f'{value:.{precision}f}'
    return str(value)


def _format_path_modes(path_modes):
    if isinstance(path_modes, (tuple, list)):
        return '|'.join(str(mode) for mode in path_modes)
    return str(path_modes)


def _to_float_or_none(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _get_model_attr(model, attr_name, default=None):
    if hasattr(model, attr_name):
        return getattr(model, attr_name)
    if hasattr(model, 'module') and hasattr(model.module, attr_name):
        return getattr(model.module, attr_name)
    return default


def _get_sp_scan_stats(model, config):
    if not getattr(config, 'use_sp_scan', False):
        return {}
    getter = _get_model_attr(model, 'get_sp_scan_stats')
    if getter is None:
        return {}
    stats = getter() or {}
    sp_cfg = getattr(config, 'sp_scan_cfg', None) or {}
    out = {
        'loss_seg': None,
        'loss_total': None,
        'sp_scan_enabled': stats.get('sp_scan_enabled', True),
        'replace_mode': stats.get('replace_mode', sp_cfg.get('replace_mode', 'NA')),
        'sp_scan_stage': stats.get('sp_scan_stage', getattr(config, 'sp_scan_stage', 'NA')),
        'num_regions_actual': stats.get('num_regions_actual'),
        'feat_h': stats.get('feat_h'),
        'feat_w': stats.get('feat_w'),
        'Q_entropy': stats.get('Q_entropy'),
        'q_max_mean': stats.get('q_max_mean'),
        'q_max_min': stats.get('q_max_min'),
        'empty_region_ratio': stats.get('empty_region_ratio'),
        'region_mass_mean': stats.get('region_mass_mean'),
        'region_mass_min': stats.get('region_mass_min'),
        'region_mass_max': stats.get('region_mass_max'),
        'region_mass_std': stats.get('region_mass_std'),
        'region_mass_nonzero_ratio': stats.get('region_mass_nonzero_ratio'),
        'region_mass_cv': stats.get('region_mass_cv'),
        'dist_margin_mean': stats.get('dist_margin_mean'),
        'logit_margin_mean': stats.get('logit_margin_mean'),
        'feat_xy_dist_ratio': stats.get('feat_xy_dist_ratio'),
        'perm_valid': stats.get('perm_valid'),
        'uses_mamba': stats.get('uses_mamba'),
        'path_types': stats.get('path_types'),
    }
    return out


def _compute_train_loss(model_out, targets, criterion, config):
    if not getattr(config, 'use_sp_rgm', False):
        loss = criterion(model_out, targets)
        return loss, {'loss_total': loss.detach()}

    out, aux = model_out
    sp_aux = aux['sp_rgm']
    loss_seg = criterion(out, targets)
    loss_region = region_label_reconstruction_loss(
        sp_aux['Q'],
        targets,
        sp_aux['feat_hw'],
        num_classes=config.num_classes,
    )
    loss_compact = region_compactness_loss(
        sp_aux['Q'],
        sp_aux['region_xy'],
        sp_aux['feat_hw'],
    )
    lambda_region = getattr(config, 'lambda_region', 0.0)
    lambda_compact = getattr(config, 'lambda_compact', 0.0)
    lambda_balance = getattr(config, 'lambda_balance', 0.0)
    if lambda_balance > 0:
        loss_balance = region_mass_balance_loss(sp_aux['Q'])
    else:
        loss_balance = loss_seg.new_tensor(0.0)
    loss_total = (
        loss_seg
        + lambda_region * loss_region
        + lambda_compact * loss_compact
        + lambda_balance * loss_balance
    )
    stats = {
        'loss_seg': loss_seg.detach(),
        'loss_region': loss_region.detach(),
        'loss_compact': loss_compact.detach(),
        'loss_balance': loss_balance.detach(),
        'loss_total': loss_total.detach(),
        'lambda_region': lambda_region,
        'lambda_compact': lambda_compact,
        'lambda_balance': lambda_balance,
        'Q_entropy': sp_aux['Q_entropy'],
        'empty_region_ratio': sp_aux['empty_region_ratio'],
        'region_mass_mean': sp_aux['region_mass_mean'],
        'region_mass_min': sp_aux['region_mass_min'],
        'region_mass_max': sp_aux['region_mass_max'],
        'region_mass_std': sp_aux['region_mass_std'],
        'region_mass_nonzero_ratio': sp_aux['region_mass_nonzero_ratio'],
        'region_mass_cv': sp_aux['region_mass_cv'],
        'q_max_mean': sp_aux['q_max_mean'],
        'q_max_min': sp_aux['q_max_min'],
        'outer_gamma': sp_aux['outer_gamma'],
        'inner_gamma': sp_aux['inner_gamma'],
        'num_regions_effective': sp_aux['num_regions_effective'],
        'num_regions_actual': sp_aux['num_regions_actual'],
        'feat_h': sp_aux['feat_h'],
        'feat_w': sp_aux['feat_w'],
        'uses_mamba': sp_aux['uses_mamba'],
        'path_modes': sp_aux['path_modes'],
        'num_paths_actual': sp_aux['num_paths_actual'],
    }
    for key in (
        'dist_min_mean',
        'dist_second_mean',
        'dist_margin_mean',
        'dist_margin_max',
        'logit_margin_mean',
        'logit_margin_max',
        'feat_dist_mean',
        'xy_dist_mean',
        'feat_xy_dist_ratio',
        'feat_margin_mean',
        'xy_margin_mean',
    ):
        if key in sp_aux:
            stats[key] = sp_aux[key]
    return loss_total, stats


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []
    sp_scan_epoch_values = {}

    for iter, data in enumerate(train_loader):
        step += 1
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images, return_aux=True) if getattr(config, 'use_sp_rgm', False) else model(images)
        loss, loss_stats = _compute_train_loss(out, targets, criterion, config)
        if getattr(config, 'use_sp_scan', False):
            sp_scan_stats = _get_sp_scan_stats(model, config)
            sp_scan_stats['loss_seg'] = loss.detach()
            sp_scan_stats['loss_total'] = loss.detach()
            loss_stats.update(sp_scan_stats)
            for key in (
                'Q_entropy',
                'q_max_mean',
                'q_max_min',
                'empty_region_ratio',
                'region_mass_mean',
                'region_mass_min',
                'region_mass_max',
                'region_mass_std',
                'region_mass_nonzero_ratio',
                'region_mass_cv',
                'dist_margin_mean',
                'logit_margin_mean',
                'feat_xy_dist_ratio',
            ):
                value = _to_float_or_none(loss_stats.get(key))
                if value is not None:
                    sp_scan_epoch_values.setdefault(key, []).append(value)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            if getattr(config, 'use_sp_rgm', False):
                log_info = (
                    f"train: epoch {epoch}, iter:{iter}, "
                    f"loss_seg: {loss_stats['loss_seg'].item():.4f}, "
                    f"loss_region: {loss_stats['loss_region'].item():.4f}, "
                    f"loss_compact: {loss_stats['loss_compact'].item():.4f}, "
                    f"loss_balance: {loss_stats['loss_balance'].item():.4f}, "
                    f"loss_total: {loss_stats['loss_total'].item():.4f}, "
                    f"Q_entropy: {loss_stats['Q_entropy'].item():.4f}, "
                    f"empty_region_ratio: {loss_stats['empty_region_ratio'].item():.4f}, "
                    f"region_mass_mean: {loss_stats['region_mass_mean'].item():.4f}, "
                    f"region_mass_min: {loss_stats['region_mass_min'].item():.4f}, "
                    f"region_mass_max: {loss_stats['region_mass_max'].item():.4f}, "
                    f"region_mass_std: {loss_stats['region_mass_std'].item():.4f}, "
                    f"region_mass_nonzero_ratio: {loss_stats['region_mass_nonzero_ratio'].item():.4f}, "
                    f"region_mass_cv: {loss_stats['region_mass_cv'].item():.4f}, "
                    f"q_max_mean: {loss_stats['q_max_mean'].item():.4f}, "
                    f"q_max_min: {loss_stats['q_max_min'].item():.4f}, "
                    f"dist_min_mean: {_format_log_value(loss_stats.get('dist_min_mean'))}, "
                    f"dist_second_mean: {_format_log_value(loss_stats.get('dist_second_mean'))}, "
                    f"dist_margin_mean: {_format_log_value(loss_stats.get('dist_margin_mean'))}, "
                    f"dist_margin_max: {_format_log_value(loss_stats.get('dist_margin_max'))}, "
                    f"logit_margin_mean: {_format_log_value(loss_stats.get('logit_margin_mean'))}, "
                    f"logit_margin_max: {_format_log_value(loss_stats.get('logit_margin_max'))}, "
                    f"feat_dist_mean: {_format_log_value(loss_stats.get('feat_dist_mean'))}, "
                    f"xy_dist_mean: {_format_log_value(loss_stats.get('xy_dist_mean'))}, "
                    f"feat_xy_dist_ratio: {_format_log_value(loss_stats.get('feat_xy_dist_ratio'))}, "
                    f"feat_margin_mean: {_format_log_value(loss_stats.get('feat_margin_mean'))}, "
                    f"xy_margin_mean: {_format_log_value(loss_stats.get('xy_margin_mean'))}, "
                    f"outer_gamma: {loss_stats['outer_gamma'].item():.6f}, "
                    f"inner_gamma: {loss_stats['inner_gamma'].item():.6f}, "
                    f"num_regions_effective: {loss_stats['num_regions_effective']}, "
                    f"num_regions_actual: {loss_stats['num_regions_actual']}, "
                    f"feat_h: {loss_stats['feat_h']}, "
                    f"feat_w: {loss_stats['feat_w']}, "
                    f"uses_mamba: {loss_stats['uses_mamba']}, "
                    f"path_modes: {_format_path_modes(loss_stats['path_modes'])}, "
                    f"num_paths_actual: {loss_stats['num_paths_actual']}, "
                    f"lambda_region: {loss_stats['lambda_region']}, "
                    f"lambda_compact: {loss_stats['lambda_compact']}, "
                    f"lambda_balance: {loss_stats['lambda_balance']}, "
                    f"lr: {now_lr}"
                )
            elif getattr(config, 'use_sp_scan', False):
                log_info = (
                    f"train: epoch {epoch}, iter:{iter}, "
                    f"loss: {_format_log_value(loss_stats.get('loss_total'))}, "
                    f"sp_scan_enabled: {_format_log_value(loss_stats.get('sp_scan_enabled'))}, "
                    f"replace_mode: {_format_log_value(loss_stats.get('replace_mode'))}, "
                    f"sp_scan_stage: {_format_log_value(loss_stats.get('sp_scan_stage'))}, "
                    f"num_regions_actual: {_format_log_value(loss_stats.get('num_regions_actual'))}, "
                    f"feat_h: {_format_log_value(loss_stats.get('feat_h'))}, "
                    f"feat_w: {_format_log_value(loss_stats.get('feat_w'))}, "
                    f"Q_entropy: {_format_log_value(loss_stats.get('Q_entropy'))}, "
                    f"q_max_mean: {_format_log_value(loss_stats.get('q_max_mean'))}, "
                    f"q_max_min: {_format_log_value(loss_stats.get('q_max_min'))}, "
                    f"empty_region_ratio: {_format_log_value(loss_stats.get('empty_region_ratio'))}, "
                    f"region_mass_mean: {_format_log_value(loss_stats.get('region_mass_mean'))}, "
                    f"region_mass_min: {_format_log_value(loss_stats.get('region_mass_min'))}, "
                    f"region_mass_max: {_format_log_value(loss_stats.get('region_mass_max'))}, "
                    f"region_mass_std: {_format_log_value(loss_stats.get('region_mass_std'))}, "
                    f"region_mass_nonzero_ratio: {_format_log_value(loss_stats.get('region_mass_nonzero_ratio'))}, "
                    f"region_mass_cv: {_format_log_value(loss_stats.get('region_mass_cv'))}, "
                    f"dist_margin_mean: {_format_log_value(loss_stats.get('dist_margin_mean'))}, "
                    f"logit_margin_mean: {_format_log_value(loss_stats.get('logit_margin_mean'))}, "
                    f"feat_xy_dist_ratio: {_format_log_value(loss_stats.get('feat_xy_dist_ratio'))}, "
                    f"perm_valid: {_format_log_value(loss_stats.get('perm_valid'))}, "
                    f"uses_mamba: {_format_log_value(loss_stats.get('uses_mamba'))}, "
                    f"path_types: {_format_path_modes(loss_stats.get('path_types'))}, "
                    f"lr: {now_lr}"
                )
            else:
                log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    if getattr(config, 'use_sp_scan', False) and sp_scan_epoch_values:
        summary = {key: float(np.mean(values)) for key, values in sp_scan_epoch_values.items()}
        log_info = (
            f"train epoch {epoch} sp_scan_mean: "
            f"Q_entropy: {_format_log_value(summary.get('Q_entropy'))}, "
            f"q_max_mean: {_format_log_value(summary.get('q_max_mean'))}, "
            f"q_max_min: {_format_log_value(summary.get('q_max_min'))}, "
            f"empty_region_ratio: {_format_log_value(summary.get('empty_region_ratio'))}, "
            f"region_mass_mean: {_format_log_value(summary.get('region_mass_mean'))}, "
            f"region_mass_min: {_format_log_value(summary.get('region_mass_min'))}, "
            f"region_mass_max: {_format_log_value(summary.get('region_mass_max'))}, "
            f"region_mass_std: {_format_log_value(summary.get('region_mass_std'))}, "
            f"region_mass_nonzero_ratio: {_format_log_value(summary.get('region_mass_nonzero_ratio'))}, "
            f"region_mass_cv: {_format_log_value(summary.get('region_mass_cv'))}, "
            f"dist_margin_mean: {_format_log_value(summary.get('dist_margin_mean'))}, "
            f"logit_margin_mean: {_format_log_value(summary.get('logit_margin_mean'))}, "
            f"feat_xy_dist_ratio: {_format_log_value(summary.get('feat_xy_dist_ratio'))}"
        )
        print(log_info)
        logger.info(log_info)

    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            if type(out) is tuple:
                out = out[0]
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    metrics = _binary_metrics_from_arrays(preds, gts, config.threshold)
    metrics['loss'] = float(np.mean(loss_list))
    # TODO: add HD95 / Boundary F1 here if those ISIC metrics are introduced later.

    log_info = _format_eval_log(f'val epoch: {epoch}', metrics)
    print(log_info)
    logger.info(log_info)
    
    return metrics


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            if type(out) is tuple:
                out = out[0]
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        metrics = _binary_metrics_from_arrays(preds, gts, config.threshold)
        metrics['loss'] = float(np.mean(loss_list))
        # TODO: add HD95 / Boundary F1 here if those ISIC metrics are introduced later.

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        checkpoint_label = getattr(config, 'active_test_checkpoint_type', 'best model')
        log_info = _format_eval_log(f'test of {checkpoint_label}', metrics)
        print(log_info)
        logger.info(log_info)

    return metrics
