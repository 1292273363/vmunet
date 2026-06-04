import torch
from torch import nn
import torch.nn.functional as F


class SoftSLIC(nn.Module):
    """Differentiable SLIC-style soft assignment implemented in pure PyTorch.

    This module follows the SSN algorithmic idea without copying its original
    Caffe/CUDA implementation so it stays easy to inspect and debug here.
    """

    def __init__(self, num_regions=(8, 8), num_iters=5, tau=0.1, xy_weight=2.0, eps=1e-6):
        super().__init__()
        self.num_regions = tuple(num_regions)
        self.num_iters = num_iters
        self.tau = tau
        self.xy_weight = xy_weight
        self.eps = eps

    def effective_num_regions(self, h, w):
        """Clamp the requested region grid to the available feature grid."""
        gh, gw = self.num_regions
        return min(gh, h), min(gw, w)

    def forward(self, feat):
        """Build soft pixel-to-region assignments from a feature map.

        Args:
            feat: Tensor[B, C, H, W]

        Returns:
            region_tokens: Tensor[B, K, C]
            Q: Tensor[B, N, K]
            recon_feat: Tensor[B, C, H, W]
            region_xy: Tensor[B, K, 2]
        """
        b, c, h, w = feat.shape
        gh, gw = self.num_regions
        gh_eff, gw_eff = self.effective_num_regions(h, w)
        n = h * w
        k = gh_eff * gw_eff

        # feat_flat: [B, N, C]
        feat_flat = feat.flatten(2).transpose(1, 2)

        # xy_grid: [1, N, 2], normalized as (x, y) in [0, 1].
        ys = torch.linspace(0.0, 1.0, h, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(0.0, 1.0, w, device=feat.device, dtype=feat.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        xy_grid = torch.stack([xx, yy], dim=-1).reshape(1, n, 2)
        xy_flat = xy_grid.expand(b, -1, -1)

        # pixel_desc: [B, N, C + 2]
        pixel_desc = torch.cat([feat_flat, xy_flat * self.xy_weight], dim=-1)
        desc_map = pixel_desc.transpose(1, 2).reshape(b, c + 2, h, w)

        # centers: [B, K, C + 2], initialized by adaptive average pooling.
        centers = F.adaptive_avg_pool2d(desc_map, output_size=(gh_eff, gw_eff))
        centers = centers.flatten(2).transpose(1, 2)

        # Q: [B, N, K], updated through differentiable soft SLIC iterations.
        for _ in range(self.num_iters):
            dist = torch.cdist(pixel_desc, centers)
            Q = torch.softmax(-dist / self.tau, dim=-1)
            center_mass = Q.sum(dim=1).unsqueeze(-1).clamp_min(self.eps)
            centers = torch.bmm(Q.transpose(1, 2), pixel_desc) / center_mass

        # region_mass: [B, K, 1], differentiable soft pixel count per region.
        region_mass = Q.sum(dim=1).unsqueeze(-1).clamp_min(self.eps)

        # region_tokens: [B, K, C], normalized pixel-to-region pooling
        # R = (Q^T X) / sum(Q), using original feature values.
        region_tokens = torch.bmm(Q.transpose(1, 2), feat_flat) / region_mass

        # recon_flat: [B, N, C], recon_feat: [B, C, H, W]
        recon_flat = torch.bmm(Q, region_tokens)
        recon_feat = recon_flat.transpose(1, 2).reshape(b, c, h, w)

        # region_xy: [B, K, 2]
        region_xy = centers[..., -2:] / self.xy_weight
        return region_tokens, Q, recon_feat, region_xy
