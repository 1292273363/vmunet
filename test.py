import torch
from tools.checkpoint_utils import extract_model_state, strip_thop_buffers, resolve_checkpoint
from utils import load_config_class

cfg = load_config_class('config_setting_isic2018_A2_sp_rgm_full_loss_4x4.py')
ckpt_path = resolve_checkpoint(cfg, 'best_loss')
ckpt = torch.load(ckpt_path, map_location='cpu')
state, _ = strip_thop_buffers(extract_model_state(ckpt))

print('checkpoint:', ckpt_path)
for name, tensor in state.items():
    if 'sp_rgm' in name:
        x = tensor.float()
        print(
            f'{name:80s} shape={tuple(tensor.shape)} '
            f'mean={x.mean().item():.6g} '
            f'std={x.std().item() if x.numel() > 1 else 0:.6g} '
            f'min={x.min().item():.6g} max={x.max().item():.6g}'
        )
