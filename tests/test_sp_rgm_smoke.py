import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from losses.region_losses import region_compactness_loss, region_label_reconstruction_loss
from models.vmunet.vmunet import VMUNet


def test_baseline_smoke():
    if not torch.cuda.is_available():
        print('skip baseline VMUNet smoke: VMamba selective_scan requires CUDA.')
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VMUNet(
        input_channels=3,
        num_classes=1,
        depths=[1, 1, 1, 1],
        depths_decoder=[1, 1, 1, 1],
        use_sp_rgm=False,
    ).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 224, 224, device=device))

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 1, 224, 224)


def test_sp_rgm_smoke():
    if not torch.cuda.is_available():
        print('skip full VMUNet SP-RGM smoke: VMamba selective_scan requires CUDA.')
        return
    model = VMUNet(
        input_channels=3,
        num_classes=1,
        depths=[1, 1, 1, 1],
        depths_decoder=[1, 1, 1, 1],
        use_sp_rgm=True,
        sp_rgm_cfg={
            'num_regions': (8, 8),
            'num_iters': 1,
            'allow_gru_fallback': True,
        },
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    x = torch.randn(2, 3, 224, 224, device=device)
    target = torch.randint(0, 2, (2, 1, 224, 224), device=device).float()

    logits, aux = model(x, return_aux=True)
    sp_aux = aux['sp_rgm']

    assert logits.shape == (2, 1, 224, 224)
    assert sp_aux['feat_hw'] == (7, 7)
    assert sp_aux['num_regions_requested'] == (8, 8)
    assert sp_aux['num_regions_effective'] == (7, 7)
    assert sp_aux['Q'].shape == (2, sp_aux['feat_hw'][0] * sp_aux['feat_hw'][1], 49)
    assert sp_aux['path_modes'] == ('yx', 'xy', 'graph', 'reverse_graph')

    loss = logits.mean()
    loss = loss + region_label_reconstruction_loss(
        sp_aux['Q'],
        target,
        sp_aux['feat_hw'],
        num_classes=1,
    )
    loss = loss + 1e-3 * region_compactness_loss(
        sp_aux['Q'],
        sp_aux['region_xy'],
        sp_aux['feat_hw'],
    )
    loss.backward()


if __name__ == '__main__':
    test_baseline_smoke()
    test_sp_rgm_smoke()
