import torch
from torch import nn

from .graph_mamba import RegionGraphMamba
from .soft_slic import SoftSLIC


class SuperpixelRegionGraphMambaBlock(nn.Module):
    """Fuse bottleneck features with superpixel region graph context."""

    def __init__(
        self,
        dim,
        num_regions=(8, 8),
        num_iters=5,
        tau=0.1,
        xy_weight=2.0,
        feat_weight=0.2,
        normalize_assign=True,
        assign_norm='layer',
        return_distance_stats=True,
        d_state=16,
        d_conv=4,
        expand=2,
        k_spatial=6,
        k_feature=6,
        alpha=0.5,
        beta=0.5,
        init_gamma=1e-3,
        path_modes=None,
        allow_gru_fallback=False,
        empty_region_eps=1e-3,
    ):
        super().__init__()
        self.empty_region_eps = empty_region_eps
        self.soft_slic = SoftSLIC(
            num_regions=num_regions,
            num_iters=num_iters,
            tau=tau,
            xy_weight=xy_weight,
            feat_weight=feat_weight,
            normalize_assign=normalize_assign,
            assign_norm=assign_norm,
            return_distance_stats=return_distance_stats,
        )
        self.region_mamba = RegionGraphMamba(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            k_spatial=k_spatial,
            k_feature=k_feature,
            alpha=alpha,
            beta=beta,
            init_gamma=init_gamma,
            path_modes=path_modes,
            allow_gru_fallback=allow_gru_fallback,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.gamma = nn.Parameter(torch.full((1,), init_gamma))

    def forward(self, x):
        """Apply the SP-RGM bottleneck block.

        Args:
            x: Tensor[B, C, H, W]

        Returns:
            out: Tensor[B, C, H, W]
            aux: dict with differentiable region tensors for training losses
        """
        b, c, h, w = x.shape
        soft_slic_out = self.soft_slic(x)
        if len(soft_slic_out) == 5:
            region_tokens, Q, recon_feat, region_xy, slic_stats = soft_slic_out
        else:
            region_tokens, Q, recon_feat, region_xy = soft_slic_out
            slic_stats = {}
        updated_tokens = self.region_mamba(region_tokens, region_xy)

        # updated_flat: [B, N, C], updated_feat: [B, C, H, W]
        updated_flat = torch.bmm(Q, updated_tokens)
        updated_feat = updated_flat.transpose(1, 2).reshape(b, c, h, w)

        fused = self.fuse(torch.cat([x, updated_feat], dim=1))
        out = x + self.gamma * fused

        # region_mass: [B, K]
        region_mass = Q.sum(dim=1)
        region_mass_std = region_mass.std(dim=1)
        region_mass_mean_per_sample = region_mass.mean(dim=1)
        q_max = Q.max(dim=-1).values
        q_entropy = -(Q.clamp_min(1e-6) * Q.clamp_min(1e-6).log()).sum(dim=-1).mean()
        empty_region_ratio = (region_mass <= self.empty_region_eps).float().mean()
        num_regions_effective = self.soft_slic.effective_num_regions(h, w)

        aux = {
            'Q': Q,
            'region_xy': region_xy,
            'region_tokens': region_tokens,
            'updated_tokens': updated_tokens,
            'recon_feat': recon_feat,
            'feat_hw': (h, w),
            'num_regions_requested': self.soft_slic.num_regions,
            'num_regions_effective': num_regions_effective,
            'num_regions_actual': region_xy.size(1),
            'feat_h': h,
            'feat_w': w,
            'Q_entropy': q_entropy.detach(),
            'empty_region_ratio': empty_region_ratio.detach(),
            'region_mass_mean': region_mass.mean().detach(),
            'region_mass_min': region_mass.min().detach(),
            'region_mass_max': region_mass.max().detach(),
            'region_mass_std': region_mass_std.mean().detach(),
            'region_mass_nonzero_ratio': (region_mass > self.empty_region_eps).float().mean().detach(),
            'region_mass_cv': (
                region_mass_std / region_mass_mean_per_sample.clamp_min(1e-6)
            ).mean().detach(),
            'q_max_mean': q_max.mean().detach(),
            'q_max_min': q_max.min().detach(),
            'outer_gamma': self.gamma.detach().clone(),
            'inner_gamma': self.region_mamba.gamma.detach().clone(),
            'uses_mamba': self.region_mamba.uses_mamba,
            'path_modes': self.region_mamba.path_modes,
            'num_paths_actual': len(self.region_mamba.path_modes),
        }
        aux.update(slic_stats)
        return out, aux
