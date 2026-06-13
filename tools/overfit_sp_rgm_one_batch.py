import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.dataset import NPY_datasets, RandomGenerator
from losses.region_losses import region_compactness_loss, region_label_reconstruction_loss
from models.vmunet.vmunet import VMUNet
from utils import get_optimizer, load_config_class, set_seed


def _load_config(module_path):
    return load_config_class(module_path)


def _enable_sp_rgm(config):
    config.use_sp_rgm = True
    config.use_sp_scan = False
    config.sp_rgm_cfg = dict(config.sp_rgm_cfg)
    config.model_config = dict(config.model_config)
    config.model_config['use_sp_rgm'] = True
    config.model_config['sp_rgm_cfg'] = config.sp_rgm_cfg
    config.model_config['use_sp_scan'] = False
    config.model_config['sp_scan_cfg'] = None
    config.model_config['sp_scan_stage'] = None


def _build_train_loader(config):
    if getattr(config, 'datasets_name', None) == 'synapse':
        dataset = config.datasets(
            base_dir=config.data_path,
            list_dir=config.list_dir,
            split='train',
            transform=transforms.Compose([
                RandomGenerator(output_size=[config.input_size_h, config.input_size_w])
            ]),
        )
    else:
        dataset = NPY_datasets(config.data_path, config, train=True)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
    )


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch['image'], batch['label']
    return batch


def _build_model(config, device):
    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
        use_sp_rgm=model_cfg['use_sp_rgm'],
        sp_rgm_cfg=model_cfg['sp_rgm_cfg'],
        use_sp_scan=model_cfg.get('use_sp_scan', getattr(config, 'use_sp_scan', False)),
        sp_scan_cfg=model_cfg.get('sp_scan_cfg', getattr(config, 'sp_scan_cfg', None)),
        sp_scan_stage=model_cfg.get('sp_scan_stage', getattr(config, 'sp_scan_stage', None)),
    )
    if model_cfg.get('load_ckpt_path') and os.path.exists(model_cfg['load_ckpt_path']):
        model.load_from()
    return model.to(device)


def _compute_losses(logits, aux, targets, criterion, config):
    sp_aux = aux['sp_rgm']
    loss_seg = criterion(logits, targets)
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
    return loss_total, loss_seg, loss_region, loss_compact


def _save_prediction(logits, output_path, num_classes):
    if num_classes == 1:
        pred = logits[0, 0].detach().cpu()
        plt.imsave(output_path, pred.numpy(), cmap='gray', vmin=0.0, vmax=1.0)
    else:
        pred = logits[0].argmax(dim=0).detach().cpu()
        plt.imsave(output_path, pred.numpy(), cmap='tab20')


def main():
    parser = argparse.ArgumentParser(description='Overfit SP-RGM on one fixed training batch.')
    parser.add_argument('--config', default='configs.config_setting')
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--output-dir', default='tools/outputs/overfit_sp_rgm_one_batch')
    args = parser.parse_args()

    config = _load_config(args.config)
    _enable_sp_rgm(config)
    set_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = _build_train_loader(config)
    images, targets = _unpack_batch(next(iter(train_loader)))
    images = images.to(device, non_blocking=True).float()
    targets = targets.to(device, non_blocking=True).float()

    model = _build_model(config, device)
    model.train()
    criterion = config.criterion
    optimizer = get_optimizer(config, model)

    for step in range(1, args.iters + 1):
        optimizer.zero_grad()
        logits, aux = model(images, return_aux=True)
        loss_total, loss_seg, loss_region, loss_compact = _compute_losses(
            logits,
            aux,
            targets,
            criterion,
            config,
        )
        loss_total.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1 or step == args.iters:
            sp_aux = aux['sp_rgm']
            print(
                f"iter {step:03d}: "
                f"loss_seg={loss_seg.item():.4f}, "
                f"loss_region={loss_region.item():.4f}, "
                f"loss_compact={loss_compact.item():.4f}, "
                f"outer_gamma={sp_aux['outer_gamma'].item():.6f}, "
                f"inner_gamma={sp_aux['inner_gamma'].item():.6f}, "
                f"Q_entropy={sp_aux['Q_entropy'].item():.4f}, "
                f"empty_region_ratio={sp_aux['empty_region_ratio'].item():.4f}"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    _save_prediction(
        logits,
        os.path.join(args.output_dir, 'prediction_final.png'),
        config.num_classes,
    )


if __name__ == '__main__':
    main()
