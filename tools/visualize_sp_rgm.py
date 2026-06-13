import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.dataset import NPY_datasets, RandomGenerator
from models.region_mamba.graph_ordering import greedy_graph_path
from models.region_mamba.region_graph import build_region_graph
from models.vmunet.vmunet import VMUNet
from tools.checkpoint_utils import load_checkpoint_state, resolve_checkpoint
from utils import load_config_class


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


def _build_dataset(config, split):
    if getattr(config, 'datasets_name', None) == 'synapse':
        transform = None
        if split == 'train':
            transform = transforms.Compose([
                RandomGenerator(output_size=[config.input_size_h, config.input_size_w])
            ])
        return config.datasets(
            base_dir=config.data_path if split == 'train' else config.volume_path,
            list_dir=config.list_dir,
            split='train' if split == 'train' else 'test_vol',
            transform=transform,
        )

    return NPY_datasets(config.data_path, config, train=(split == 'train'))


def _unpack_sample(sample):
    if isinstance(sample, dict):
        return sample['image'], sample['label']
    return sample


def _build_model(config, checkpoint, device):
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

    if checkpoint is not None:
        checkpoint_path = resolve_checkpoint(config, checkpoint)
        state = torch.load(checkpoint_path, map_location='cpu')
        load_checkpoint_state(model, state, strict=True)
    elif model_cfg.get('load_ckpt_path') and os.path.exists(model_cfg['load_ckpt_path']):
        model.load_from()

    return model.to(device).eval()


def _to_display_image(image):
    image = image.detach().cpu().float()
    if image.dim() == 3 and image.size(0) == 1:
        array = image[0].numpy()
        return array, 'gray'

    if image.dim() == 3:
        array = image.permute(1, 2, 0).numpy()
        array = array - array.min()
        array = array / max(array.max(), 1e-6)
        return array, None

    return image.numpy(), 'gray'


def _save_array(path, array, cmap=None, vmin=None, vmax=None):
    plt.imsave(path, array, cmap=cmap, vmin=vmin, vmax=vmax)


def _save_graph_path(path, region_tokens, region_xy):
    A = build_region_graph(region_tokens, region_xy)
    order = greedy_graph_path(A, region_xy)[0].detach().cpu()
    xy = region_xy[0].detach().cpu()
    ordered_xy = xy[order]

    plt.figure(figsize=(6, 6))
    plt.plot(ordered_xy[:, 0], ordered_xy[:, 1], linewidth=1.5)
    plt.scatter(xy[:, 0], xy[:, 1], s=18)
    for step, region_idx in enumerate(order.tolist()):
        plt.text(xy[region_idx, 0], xy[region_idx, 1], str(step), fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize SP-RGM predictions and region assignments.')
    parser.add_argument('--config', default='configs.config_setting')
    parser.add_argument('--checkpoint')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--output-dir', default='tools/outputs/visualize_sp_rgm')
    parser.add_argument('--save-path-order', action='store_true')
    args = parser.parse_args()

    config = _load_config(args.config)
    _enable_sp_rgm(config)

    dataset = _build_dataset(config, args.split)
    image, target = _unpack_sample(dataset[args.index])
    image_batch = image.unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _build_model(config, args.checkpoint, device)

    with torch.no_grad():
        logits, aux = model(image_batch.to(device).float(), return_aux=True)

    os.makedirs(args.output_dir, exist_ok=True)
    image_array, image_cmap = _to_display_image(image)
    _save_array(os.path.join(args.output_dir, 'input.png'), image_array, cmap=image_cmap)

    target_map = target.squeeze(0).detach().cpu()
    if config.num_classes == 1:
        pred_map = (logits[0, 0] >= config.threshold).detach().cpu().long()
        gt_map = (target_map >= 0.5).long()
        pred_to_save = pred_map.numpy()
    else:
        pred_map = logits[0].argmax(dim=0).detach().cpu()
        gt_map = target_map.long()
        pred_to_save = pred_map.numpy()

    _save_array(os.path.join(args.output_dir, 'gt_mask.png'), gt_map.numpy(), cmap='gray')
    _save_array(
        os.path.join(args.output_dir, 'prediction.png'),
        pred_to_save,
        cmap='gray' if config.num_classes == 1 else 'tab20',
    )

    error_map = (pred_map != gt_map).float().numpy()
    _save_array(os.path.join(args.output_dir, 'error_map.png'), error_map, cmap='magma', vmin=0.0, vmax=1.0)

    sp_aux = aux['sp_rgm']
    h, w = sp_aux['feat_hw']
    hard_region_map = sp_aux['Q'][0].argmax(dim=-1).reshape(1, 1, h, w).float()
    hard_region_map = F.interpolate(
        hard_region_map,
        size=gt_map.shape[-2:],
        mode='nearest',
    )[0, 0].detach().cpu().numpy()
    _save_array(os.path.join(args.output_dir, 'hard_region_map.png'), hard_region_map, cmap='tab20')

    if args.save_path_order:
        _save_graph_path(
            os.path.join(args.output_dir, 'graph_path_order.png'),
            sp_aux['region_tokens'],
            sp_aux['region_xy'],
        )


if __name__ == '__main__':
    main()
