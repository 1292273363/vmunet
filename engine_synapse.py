import numpy as np
from tqdm import tqdm

from torch.cuda.amp import autocast as autocast
import torch

from sklearn.metrics import confusion_matrix

from scipy.ndimage.morphology import binary_fill_holes, binary_opening

from utils import test_single_volume
from losses.region_losses import region_compactness_loss, region_label_reconstruction_loss

import time


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
    loss_total = loss_seg + config.lambda_region * loss_region + config.lambda_compact * loss_compact
    stats = {
        'loss_seg': loss_seg.detach(),
        'loss_region': loss_region.detach(),
        'loss_compact': loss_compact.detach(),
        'loss_total': loss_total.detach(),
        'Q_entropy': sp_aux['Q_entropy'],
        'empty_region_ratio': sp_aux['empty_region_ratio'],
        'outer_gamma': sp_aux['outer_gamma'],
        'inner_gamma': sp_aux['inner_gamma'],
    }
    return loss_total, stats


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    stime = time.time()
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()

        images, targets = data['image'], data['label']
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()   

        if config.amp:
            with autocast():
                out = model(images, return_aux=True) if getattr(config, 'use_sp_rgm', False) else model(images)
                loss, loss_stats = _compute_train_loss(out, targets, criterion, config)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images, return_aux=True) if getattr(config, 'use_sp_rgm', False) else model(images)
            loss, loss_stats = _compute_train_loss(out, targets, criterion, config)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_loss = np.mean(loss_list)
        if iter % config.print_interval == 0:
            if getattr(config, 'use_sp_rgm', False):
                log_info = (
                    f"train: epoch {epoch}, iter:{iter}, "
                    f"loss_seg: {loss_stats['loss_seg'].item():.4f}, "
                    f"loss_region: {loss_stats['loss_region'].item():.4f}, "
                    f"loss_compact: {loss_stats['loss_compact'].item():.4f}, "
                    f"loss_total: {loss_stats['loss_total'].item():.4f}, "
                    f"Q_entropy: {loss_stats['Q_entropy'].item():.4f}, "
                    f"empty_region_ratio: {loss_stats['empty_region_ratio'].item():.4f}, "
                    f"outer_gamma: {loss_stats['outer_gamma'].item():.6f}, "
                    f"inner_gamma: {loss_stats['inner_gamma'].item():.6f}, "
                    f"lr: {now_lr}"
                )
            else:
                log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    return mean_loss





def val_one_epoch(test_datasets,
                    test_loader,
                    model,
                    epoch, 
                    logger,
                    config,
                    test_save_path,
                    val_or_test=False):
    # switch to evaluate mode
    stime = time.time()
    model.eval()
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            metric_i = test_single_volume(img, msk, model, classes=config.num_classes, patch_size=[config.input_size_h, config.input_size_w],
                                    test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing, val_or_test=val_or_test)
            metric_list += np.array(metric_i)

            logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            i_batch += 1
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        for i in range(1, config.num_classes):
            logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}, time(s): {etime-stime:.2f}'
        print(log_info)
        logger.info(log_info)
    
    return performance, mean_hd95
