import math
import time

import torch
from torch import nn

from .soft_slic import SoftSLIC


def make_norm2d(num_channels, norm_type='group', num_groups=8):
    """Build a 2D normalization layer for small-batch segmentation features."""
    if norm_type == 'group':
        groups = int(num_groups)
        if groups <= 0:
            raise ValueError('num_groups must be positive when norm_type="group".')
        if num_channels % groups != 0:
            groups = 1
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels)
    if norm_type == 'batch':
        return nn.BatchNorm2d(num_channels)
    if norm_type == 'identity':
        return nn.Identity()
    raise ValueError("norm_type must be one of ['group', 'batch', 'identity'].")


def make_coord_grid(batch_size, height, width, device, dtype):
    """Return normalized coordinate features with shape [B, 2, H, W]."""
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    coord = torch.stack([xx, yy], dim=0).unsqueeze(0)
    return coord.expand(batch_size, -1, -1, -1)


def region_soft_avg_pool(feat_flat, Q, eps=1e-6):
    """Soft pixel-to-region average pooling.

    Args:
        feat_flat: Tensor[B, N, C]
        Q: Tensor[B, N, K]

    Returns:
        z_avg: Tensor[B, K, C]
        region_mass: Tensor[B, K]
    """
    region_mass = Q.sum(dim=1).clamp_min(eps)
    z_avg = torch.bmm(Q.transpose(1, 2), feat_flat) / region_mass.unsqueeze(-1)
    return z_avg, region_mass


def region_hard_max_pool(feat_flat, Q, z_avg):
    """Hard-label region max pooling with average-token fallback.

    Args:
        feat_flat: Tensor[B, N, C]
        Q: Tensor[B, N, K]
        z_avg: Tensor[B, K, C], used when a hard region is empty.

    Returns:
        z_max: Tensor[B, K, C]
    """
    labels = Q.argmax(dim=-1)
    batch_size, _, _ = feat_flat.shape
    num_regions = Q.size(-1)
    fill_value = torch.finfo(feat_flat.dtype).min
    z_max = []

    for region_idx in range(num_regions):
        mask = labels == region_idx
        region_values = feat_flat.masked_fill(~mask.unsqueeze(-1), fill_value)
        max_values = region_values.max(dim=1).values
        has_region = mask.any(dim=1).view(batch_size, 1)
        max_values = torch.where(has_region, max_values, z_avg[:, region_idx])
        z_max.append(max_values)

    return torch.stack(z_max, dim=1)


class SuperpixelSkipRefine(nn.Module):
    """Grid-preserving superpixel aggregation for VM-UNet skip features.

    This module follows the SuiT-style idea of forming temporary superpixel
    tokens from feature embeddings, then projecting them back to the original
    H x W grid. It does not replace VM-UNet patch tokens or alter SS2D scan
    order.
    """

    def __init__(
        self,
        dim,
        num_regions=(4, 4),
        num_iters=5,
        tau=0.2,
        xy_weight=2.0,
        feat_weight=0.1,
        normalize_assign=True,
        assign_norm='layer',
        use_pos_embed=True,
        use_avg_pool=True,
        use_max_pool=True,
        use_graph=False,
        region_update='mlp',
        gamma_init=1e-3,
        gate_type='bounded_tanh',
        gate_scale=0.1,
        norm_type='group',
        num_groups=8,
        detach_assignment=False,
        debug_stats=True,
        stage_name='stage',
        eps=1e-6,
    ):
        super().__init__()
        if not use_avg_pool and not use_max_pool:
            raise ValueError('SuperpixelSkipRefine requires at least one of avg or max pooling.')
        if use_graph:
            raise ValueError('G1/G2 SSR does not support graph update; set use_graph=False.')
        if region_update != 'mlp':
            raise ValueError("G1/G2 SSR only supports region_update='mlp'.")
        if gate_type not in {'bounded_tanh', 'raw'}:
            raise ValueError("gate_type must be 'bounded_tanh' or 'raw'.")

        self.dim = dim
        self.num_regions = tuple(num_regions)
        self.use_pos_embed = use_pos_embed
        self.use_avg_pool = use_avg_pool
        self.use_max_pool = use_max_pool
        self.use_graph = use_graph
        self.region_update = region_update
        self.gate_type = gate_type
        self.gate_scale = float(gate_scale)
        self.norm_type = norm_type
        self.num_groups = int(num_groups)
        self.detach_assignment = detach_assignment
        self.debug_stats = debug_stats
        self.stage_name = stage_name
        self.eps = eps

        self.local_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            make_norm2d(dim, norm_type=norm_type, num_groups=num_groups),
            nn.GELU(),
        )
        coord_channels = 2 if use_pos_embed else 0
        self.pixel_proj = nn.Sequential(
            nn.Conv2d(dim + coord_channels, dim, kernel_size=1, bias=False),
            make_norm2d(dim, norm_type=norm_type, num_groups=num_groups),
            nn.GELU(),
        )
        self.soft_slic = SoftSLIC(
            num_regions=num_regions,
            num_iters=num_iters,
            tau=tau,
            xy_weight=xy_weight,
            feat_weight=feat_weight,
            normalize_assign=normalize_assign,
            assign_norm=assign_norm,
            return_distance_stats=debug_stats,
            eps=eps,
        )

        pooled_dim = dim * int(use_avg_pool) + dim * int(use_max_pool)
        self.region_proj = nn.Sequential(
            nn.Linear(pooled_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.region_norm = nn.LayerNorm(dim)
        self.region_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            make_norm2d(dim, norm_type=norm_type, num_groups=num_groups),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        if gate_type == 'bounded_tanh':
            ratio = max(min(float(gamma_init) / self.gate_scale, 0.999), -0.999)
            gamma_raw_init = math.atanh(ratio)
        else:
            gamma_raw_init = float(gamma_init)
        self.gamma_raw = nn.Parameter(torch.tensor(gamma_raw_init, dtype=torch.float32))
        self.last_stats = None

    def _gate_value(self):
        if self.gate_type == 'bounded_tanh':
            return self.gate_scale * torch.tanh(self.gamma_raw)
        return self.gamma_raw

    def _make_stats(
        self,
        x,
        Q,
        region_mass,
        z_avg,
        z_max,
        recon,
        fused,
        out,
        slic_stats,
        forward_time_ms,
    ):
        q_max = Q.max(dim=-1).values
        entropy = -(Q.clamp_min(self.eps) * Q.clamp_min(self.eps).log()).sum(dim=-1).mean()
        region_mass_std = region_mass.std(dim=1).mean()
        region_mass_mean_per_batch = region_mass.mean(dim=1).clamp_min(self.eps)
        gate_value = self._gate_value()
        input_norm = x.detach().norm()
        recon_norm = recon.detach().norm()
        fused_norm = fused.detach().norm()
        delta_norm = (out.detach() - x.detach()).norm()

        stats = {
            'ssr_enabled': True,
            'ssr_stage': self.stage_name,
            'ssr_enabled_stage': self.stage_name,
            'ssr_feature_shape': tuple(x.shape),
            'ssr_num_regions': self.num_regions,
            'num_regions': self.num_regions,
            'num_regions_actual': Q.shape[-1],
            'feat_h': x.shape[-2],
            'feat_w': x.shape[-1],
            'Q_entropy': entropy.detach(),
            'q_max_mean': q_max.mean().detach(),
            'q_max_min': q_max.min().detach(),
            'empty_region_ratio': (region_mass <= self.eps).float().mean().detach(),
            'region_mass_mean': region_mass.mean().detach(),
            'region_mass_min': region_mass.min().detach(),
            'region_mass_max': region_mass.max().detach(),
            'region_mass_std': region_mass_std.detach(),
            'region_mass_nonzero_ratio': (region_mass > self.eps).float().mean().detach(),
            'region_mass_cv': (region_mass.std(dim=1) / region_mass_mean_per_batch).mean().detach(),
            'gate_value': gate_value.detach(),
            'gamma_raw': self.gamma_raw.detach(),
            'gate_scale': self.gate_scale,
            'gate_type': self.gate_type,
            'norm_type': self.norm_type,
            'num_groups': self.num_groups,
            'detach_assignment': self.detach_assignment,
            'use_avg_pool': self.use_avg_pool,
            'use_max_pool': self.use_max_pool,
            'use_pos_embed': self.use_pos_embed,
            'use_graph': self.use_graph,
            'region_update': self.region_update,
            'input_norm': input_norm,
            'recon_norm': recon_norm,
            'fused_norm': fused_norm,
            'recon_input_norm_ratio': recon_norm / input_norm.clamp_min(self.eps),
            'output_input_delta_norm': delta_norm,
            'output_input_delta_ratio': delta_norm / input_norm.clamp_min(self.eps),
            'ssr_forward_time_ms': forward_time_ms,
            'Q_has_nan': torch.isnan(Q).any().detach(),
            'z_avg_has_nan': torch.isnan(z_avg).any().detach(),
            'z_max_has_nan': torch.isnan(z_max).any().detach(),
        }
        for key in (
            'dist_margin_mean',
            'logit_margin_mean',
            'feat_xy_dist_ratio',
            'dist_min_mean',
            'dist_second_mean',
            'feat_dist_mean',
            'xy_dist_mean',
        ):
            if key in slic_stats:
                value = slic_stats[key]
                stats[key] = value.detach() if torch.is_tensor(value) else value
        return stats

    def forward(self, x, return_stats=False):
        """Refine a skip feature map without changing its grid shape.

        Args:
            x: Tensor[B, C, H, W]

        Returns:
            out: Tensor[B, C, H, W]
            stats: dict, only when return_stats=True
        """
        start_time = time.perf_counter()
        b, c, h, w = x.shape
        pixel_embed = self.local_embed(x)
        if self.use_pos_embed:
            coord = make_coord_grid(b, h, w, x.device, x.dtype)
            pixel_input = torch.cat([pixel_embed, coord], dim=1)
        else:
            pixel_input = pixel_embed
        pixel_feat = self.pixel_proj(pixel_input)

        slic_input = pixel_feat.detach() if self.detach_assignment else pixel_feat
        slic_out = self.soft_slic(slic_input)
        if len(slic_out) == 5:
            _, Q, _, _, slic_stats = slic_out
        else:
            _, Q, _, _ = slic_out
            slic_stats = {}

        # feat_flat: [B, N, C], Q: [B, N, K]
        feat_flat = pixel_feat.flatten(2).transpose(1, 2)
        z_avg, region_mass = region_soft_avg_pool(feat_flat, Q, eps=self.eps)
        pooled = []
        if self.use_avg_pool:
            pooled.append(z_avg)
        if self.use_max_pool:
            z_max = region_hard_max_pool(feat_flat, Q, z_avg)
            pooled.append(z_max)
        else:
            z_max = z_avg

        # z_refined: [B, K, C]
        z = self.region_proj(torch.cat(pooled, dim=-1))
        z_refined = z + self.region_mlp(self.region_norm(z))

        # recon: [B, C, H, W], grid-preserving region-to-pixel projection.
        recon_flat = torch.bmm(Q, z_refined)
        recon = recon_flat.transpose(1, 2).reshape(b, c, h, w)
        fused = self.fuse(torch.cat([x, recon], dim=1))
        out = x + self._gate_value().to(dtype=x.dtype, device=x.device) * fused

        forward_time_ms = (time.perf_counter() - start_time) * 1000.0
        stats = self._make_stats(x, Q, region_mass, z_avg, z_max, recon, fused, out, slic_stats, forward_time_ms)
        self.last_stats = stats
        if return_stats:
            return out, stats
        return out
