import glob
import os
from collections import OrderedDict


CHECKPOINT_ALIASES = {'best_loss', 'best_dice', 'best_iou', 'latest'}
THOP_BUFFER_NAMES = {'total_ops', 'total_params'}


def is_thop_buffer_key(key):
    """Return True for THOP profiling buffers saved inside a state dict."""
    return key.split('.')[-1] in THOP_BUFFER_NAMES


def strip_thop_buffers(state_dict):
    """Remove THOP total_ops/total_params buffers from a model state dict."""
    if not isinstance(state_dict, dict):
        return state_dict, []

    cleaned = OrderedDict()
    removed = []
    for key, value in state_dict.items():
        if is_thop_buffer_key(key):
            removed.append(key)
            continue
        cleaned[key] = value
    return cleaned, removed


def extract_model_state(checkpoint):
    """Extract model weights from either raw or structured checkpoints."""
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'model' in checkpoint:
            return checkpoint['model']
    return checkpoint


def load_checkpoint_state(model, checkpoint, strict=True):
    """Load checkpoint weights into a model while ignoring THOP profiling buffers."""
    state_dict = extract_model_state(checkpoint)
    state_dict, removed_keys = strip_thop_buffers(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return incompatible, removed_keys


def available_checkpoints(checkpoint_dir):
    """List checkpoint files under a checkpoint directory."""
    if not os.path.isdir(checkpoint_dir):
        return []
    return sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth')))


def resolve_checkpoint(config, ckpt):
    """Resolve an alias or path to an existing checkpoint file.

    Supports the current names best_loss/best_dice/best_iou/latest and the
    legacy VM-UNet best-epoch*-loss*.pth name used by earlier A0/A1 runs.
    """
    if ckpt is None:
        return None

    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    if ckpt in CHECKPOINT_ALIASES:
        candidate = os.path.join(checkpoint_dir, f'{ckpt}.pth')
        if os.path.exists(candidate):
            return candidate

        if ckpt == 'best_loss':
            legacy = sorted(glob.glob(os.path.join(checkpoint_dir, 'best-epoch*-loss*.pth')))
            if legacy:
                return legacy[-1]

        existing = available_checkpoints(checkpoint_dir)
        raise FileNotFoundError(
            f"Checkpoint alias '{ckpt}' was not found under {checkpoint_dir}. "
            f"Available checkpoints: {existing}"
        )

    if os.path.exists(ckpt):
        return ckpt

    candidate = os.path.join(checkpoint_dir, ckpt)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt}. Available checkpoints: "
        f"{available_checkpoints(checkpoint_dir)}"
    )
