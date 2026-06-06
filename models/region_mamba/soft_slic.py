import torch
from torch import nn
import torch.nn.functional as F


class SoftSLIC(nn.Module):
    """Differentiable SLIC-style soft assignment implemented in pure PyTorch.

    This module follows the SSN algorithmic idea without copying its original
    Caffe/CUDA implementation so it stays easy to inspect and debug here.
    """

    def __init__(
        self,
        num_regions=(8, 8),
        num_iters=5,
        tau=0.1,
        xy_weight=2.0,
        feat_weight=0.2,
        normalize_assign=True,
        assign_norm='layer',
        return_distance_stats=True,
        eps=1e-6,
    ):
        super().__init__()
        self.num_regions = tuple(num_regions)
        self.num_iters = num_iters
        self.tau = tau
        self.xy_weight = xy_weight
        self.feat_weight = feat_weight
        self.normalize_assign = normalize_assign
        self.assign_norm = assign_norm
        self.return_distance_stats = return_distance_stats
        self.eps = eps

    def effective_num_regions(self, h, w):
        """Clamp the requested region grid to the available feature grid."""
        gh, gw = self.num_regions
        return min(gh, h), min(gw, w)

    def _normalize_assignment_feature(self, feat_flat, channels):
        # feat_flat: [B, N, C], returned tensor keeps [B, N, C].
        if not self.normalize_assign or self.assign_norm == 'none':
            return feat_flat
        if self.assign_norm == 'layer':
            return F.layer_norm(feat_flat, (channels,))
        if self.assign_norm == 'l2':
            return F.normalize(feat_flat, dim=-1, eps=self.eps)
        raise ValueError(
            f"Unknown SoftSLIC assign_norm: {self.assign_norm}. "
            "Supported values are ['layer', 'l2', 'none']."
        )

    def _distance_stats(self, dist, pixel_desc, centers, assign_channels):
        # dist: [B, N, K], pixel_desc: [B, N, C_assign + 2], centers: [B, K, C_assign + 2]
        if not self.return_distance_stats:
            return {}

        dist_sorted = torch.sort(dist.detach(), dim=-1).values
        d1 = dist_sorted[..., 0]
        d2 = dist_sorted[..., 1] if dist_sorted.size(-1) > 1 else dist_sorted[..., 0]
        margin = d2 - d1
        logit_margin = margin / max(self.tau, self.eps)

        center_feat = centers[..., :assign_channels]
        center_xy = centers[..., -2:]
        feat_part = pixel_desc[..., :-2]
        xy_part = pixel_desc[..., -2:]

        feat_dist = torch.cdist(feat_part.detach(), center_feat.detach())
        xy_dist = torch.cdist(xy_part.detach(), center_xy.detach())
        feat_dist_mean = feat_dist.mean().detach()
        xy_dist_mean = xy_dist.mean().detach()

        if dist_sorted.size(-1) > 1:
            feat_dist_sorted = torch.sort(feat_dist, dim=-1).values
            xy_dist_sorted = torch.sort(xy_dist, dim=-1).values
            feat_margin_mean = (feat_dist_sorted[..., 1] - feat_dist_sorted[..., 0]).mean().detach()
            xy_margin_mean = (xy_dist_sorted[..., 1] - xy_dist_sorted[..., 0]).mean().detach()
        else:
            feat_margin_mean = feat_dist_mean.new_tensor(0.0)
            xy_margin_mean = xy_dist_mean.new_tensor(0.0)

        return {
            'dist_min_mean': d1.mean().detach(),
            'dist_second_mean': d2.mean().detach(),
            'dist_margin_mean': margin.mean().detach(),
            'dist_margin_max': margin.max().detach(),
            'logit_margin_mean': logit_margin.mean().detach(),
            'logit_margin_max': logit_margin.max().detach(),
            'feat_dist_mean': feat_dist_mean,
            'xy_dist_mean': xy_dist_mean,
            'feat_xy_dist_ratio': feat_dist_mean / xy_dist_mean.clamp_min(self.eps),
            'feat_margin_mean': feat_margin_mean,
            'xy_margin_mean': xy_margin_mean,
        }

    def forward(self, feat):
        """Build soft pixel-to-region assignments from a feature map.

        Args:
            feat: Tensor[B, C, H, W]

        Returns:
            region_tokens: Tensor[B, K, C]
            Q: Tensor[B, N, K]
            recon_feat: Tensor[B, C, H, W]
            region_xy: Tensor[B, K, 2]
            slic_stats: dict of assignment distance diagnostics, when enabled
        """
        b, c, h, w = feat.shape
        gh, gw = self.num_regions
        gh_eff, gw_eff = self.effective_num_regions(h, w)
        n = h * w
        k = gh_eff * gw_eff

        # feat_flat: [B, N, C]
        feat_flat = feat.flatten(2).transpose(1, 2)
        feat_assign = self._normalize_assignment_feature(feat_flat, c)

        # xy_grid: [1, N, 2], normalized as (x, y) in [0, 1].
        ys = torch.linspace(0.0, 1.0, h, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(0.0, 1.0, w, device=feat.device, dtype=feat.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        xy_grid = torch.stack([xx, yy], dim=-1).reshape(1, n, 2)
        xy_flat = xy_grid.expand(b, -1, -1)

        # pixel_desc: [B, N, C + 2], used only for assignment.
        pixel_desc = torch.cat([
            feat_assign * self.feat_weight,
            xy_flat * self.xy_weight,
        ], dim=-1)
        assign_channels = feat_assign.size(-1)
        desc_map = pixel_desc.transpose(1, 2).reshape(b, assign_channels + 2, h, w)

        # centers: [B, K, C + 2], initialized by adaptive average pooling.
        centers = F.adaptive_avg_pool2d(desc_map, output_size=(gh_eff, gw_eff))
        centers = centers.flatten(2).transpose(1, 2)

        # Q: [B, N, K], updated through differentiable soft SLIC iterations.
        for _ in range(self.num_iters):
            dist = torch.cdist(pixel_desc, centers)
            Q = torch.softmax(-dist / self.tau, dim=-1)
            center_mass = Q.sum(dim=1).unsqueeze(-1).clamp_min(self.eps)
            centers = torch.bmm(Q.transpose(1, 2), pixel_desc) / center_mass
        slic_stats = self._distance_stats(dist, pixel_desc, centers, assign_channels)

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
        if self.return_distance_stats:
            return region_tokens, Q, recon_feat, region_xy, slic_stats
        return region_tokens, Q, recon_feat, region_xy
