import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not available in the current Python environment. "
        "Activate the training environment first, e.g. `conda activate vmunet`, "
        "or run `/home/chenzhuoshao/miniconda3/envs/vmunet/bin/python tools/inspect_checkpoint.py ...`."
    ) from exc

from tools.checkpoint_utils import extract_model_state, resolve_checkpoint, strip_thop_buffers
from utils import load_config_class


METRIC_KEYS = [
    'checkpoint_type',
    'epoch',
    'loss',
    'dice',
    'iou',
    'jaccard',
    'accuracy',
    'specificity',
    'sensitivity',
    'confusion_matrix',
    'min_loss',
    'min_epoch',
]


def _format_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def _print_checkpoint_summary(path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = extract_model_state(checkpoint)
    cleaned_state, removed_keys = strip_thop_buffers(state_dict)
    tensor_items = [
        (name, tensor)
        for name, tensor in cleaned_state.items()
        if hasattr(tensor, 'shape')
    ] if isinstance(cleaned_state, dict) else []

    print(f'path: {path}')
    print(f'size_mb: {os.path.getsize(path) / (1024 * 1024):.2f}')
    print(f'checkpoint_type_python: {type(checkpoint).__name__}')

    if isinstance(checkpoint, dict):
        top_keys = list(checkpoint.keys())
        if 'model_state_dict' in checkpoint or any(key in checkpoint for key in METRIC_KEYS):
            print(f'top_level_keys: {top_keys}')
        else:
            print('top_level_keys: raw_state_dict')
            print(f'raw_state_key_count: {len(top_keys)}')
            print(f'raw_state_key_examples: {top_keys[:12]}')
        for key in METRIC_KEYS:
            if key in checkpoint:
                print(f'{key}: {_format_value(checkpoint[key])}')
        if 'best_records' in checkpoint:
            print(f"best_records: {checkpoint['best_records']}")
    else:
        print('top_level_keys: raw_state_dict')

    print(f'model_tensor_count: {len(tensor_items)}')
    print(f'thop_keys_removed_for_loading: {len(removed_keys)}')
    if removed_keys:
        print(f'thop_key_examples: {removed_keys[:8]}')

    print('first_tensors:')
    for name, tensor in tensor_items[:12]:
        print(f'  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}')


def main():
    parser = argparse.ArgumentParser(description='Inspect a VM-UNet checkpoint without opening the binary .pth file.')
    parser.add_argument(
        '--config',
        default=None,
        help='Config module/filename. Required when --ckpt is an alias such as best_loss.',
    )
    parser.add_argument(
        '--ckpt',
        '--checkpoint',
        required=True,
        dest='ckpt',
        help='Checkpoint path or one of best_loss/best_dice/best_iou/latest when --config is provided.',
    )
    args = parser.parse_args()

    if args.ckpt in {'best_loss', 'best_dice', 'best_iou', 'latest'}:
        if args.config is None:
            raise ValueError('--config is required when --ckpt is a checkpoint alias.')
        config = load_config_class(args.config)
        checkpoint_path = resolve_checkpoint(config, args.ckpt)
    elif args.config is not None:
        config = load_config_class(args.config)
        checkpoint_path = resolve_checkpoint(config, args.ckpt)
    else:
        checkpoint_path = args.ckpt

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    _print_checkpoint_summary(checkpoint_path)


if __name__ == '__main__':
    main()
