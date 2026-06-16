import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

from datasets.dataset import NPY_datasets
from models.vmunet.vmunet import VMUNet
from tools.checkpoint_utils import load_checkpoint_state, resolve_checkpoint
from utils import load_config_class, set_seed


def _build_model(config, checkpoint_path, device):
    model_cfg = config.model_config
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_checkpoint_state(model, checkpoint, strict=True)
    return model.to(device).eval()


def _collect_predictions(loader, model, device):
    preds = []
    gts = []
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device, non_blocking=True).float()
            out = model(images)
            if type(out) is tuple:
                out = out[0]
            preds.append(out.squeeze(1).detach().cpu().numpy())
            gts.append(targets.squeeze(1).detach().cpu().numpy())
    return np.array(preds).reshape(-1), np.array(gts).reshape(-1)


def _metrics_at_threshold(preds, gts, threshold):
    y_pred = (preds >= threshold).astype(np.uint8)
    y_true = (gts >= 0.5).astype(np.uint8)

    TP = int(np.logical_and(y_pred == 1, y_true == 1).sum())
    FP = int(np.logical_and(y_pred == 1, y_true == 0).sum())
    TN = int(np.logical_and(y_pred == 0, y_true == 0).sum())
    FN = int(np.logical_and(y_pred == 0, y_true == 1).sum())

    dice = float(2 * TP) / float(2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
    iou = float(TP) / float(TP + FP + FN) if (TP + FP + FN) else 0.0
    accuracy = float(TP + TN) / float(TP + FP + TN + FN) if (TP + FP + TN + FN) else 0.0
    specificity = float(TN) / float(TN + FP) if (TN + FP) else 0.0
    sensitivity = float(TP) / float(TP + FN) if (TP + FN) else 0.0

    return {
        'threshold': threshold,
        'Dice': dice,
        'IoU': iou,
        'Jaccard': iou,
        'Accuracy': accuracy,
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
    }


def main():
    parser = argparse.ArgumentParser(description='Threshold sweep for ISIC2018 A-group checkpoints.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True, help='Path or one of best_loss/best_dice/best_iou/latest.')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument(
        '--thresholds',
        default='0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7',
    )
    parser.add_argument('--output-csv', default=None)
    args = parser.parse_args()

    config = load_config_class(args.config)
    set_seed(config.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    checkpoint_path = resolve_checkpoint(config, args.ckpt)

    dataset = NPY_datasets(config.data_path, config, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _build_model(config, checkpoint_path, device)
    preds, gts = _collect_predictions(loader, model, device)

    thresholds = [float(x.strip()) for x in args.thresholds.split(',') if x.strip()]
    rows = [_metrics_at_threshold(preds, gts, threshold) for threshold in thresholds]

    if args.output_csv is None:
        ckpt_stem = Path(checkpoint_path).stem
        output_csv = os.path.join(config.work_dir, f'threshold_sweep_{ckpt_stem}.csv')
    else:
        output_csv = args.output_csv
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)

    fieldnames = [
        'threshold',
        'Dice',
        'IoU',
        'Jaccard',
        'Accuracy',
        'Specificity',
        'Sensitivity',
        'TP',
        'FP',
        'TN',
        'FN',
    ]
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"thr={row['threshold']:.2f} "
            f"Dice={row['Dice']:.4f} IoU={row['IoU']:.4f} "
            f"Acc={row['Accuracy']:.4f} Spec={row['Specificity']:.4f} "
            f"Sens={row['Sensitivity']:.4f} TP={row['TP']} FP={row['FP']} "
            f"TN={row['TN']} FN={row['FN']}"
        )
    print(f'saved: {output_csv}')


if __name__ == '__main__':
    main()
