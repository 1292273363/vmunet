import torch

from .graph_ordering import batched_gather_tokens, greedy_graph_path, invert_permutation
from .region_graph import build_region_graph


def _scalar_detach(value):
    if torch.is_tensor(value):
        return value.detach()
    return value


def _compute_assignment_stats(Q, feat_h, feat_w, slic_stats, eps):
    # Q: [B, L, K], region_mass: [B, K]
    region_mass = Q.sum(dim=1)
    q_max = Q.max(dim=-1).values
    entropy = -(Q.clamp_min(eps) * Q.clamp_min(eps).log()).sum(dim=-1).mean()
    mass_mean = region_mass.mean(dim=1)
    mass_std = region_mass.std(dim=1)

    stats = {
        'Q_entropy': entropy.detach(),
        'q_max_mean': q_max.mean().detach(),
        'q_max_min': q_max.min().detach(),
        'empty_region_ratio': (region_mass <= eps).float().mean().detach(),
        'region_mass_mean': region_mass.mean().detach(),
        'region_mass_min': region_mass.min().detach(),
        'region_mass_max': region_mass.max().detach(),
        'region_mass_std': mass_std.mean().detach(),
        'region_mass_nonzero_ratio': (region_mass > eps).float().mean().detach(),
        'region_mass_cv': (mass_std / mass_mean.clamp_min(eps)).mean().detach(),
        'num_regions_actual': int(Q.shape[-1]),
        'feat_h': int(feat_h),
        'feat_w': int(feat_w),
        'sp_scan_enabled': True,
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
            stats[key] = _scalar_detach(slic_stats[key])
    return stats


def _permutation_is_valid(perm):
    # perm: [B, L]
    expected = torch.arange(perm.size(1), device=perm.device).unsqueeze(0).expand_as(perm)
    return torch.equal(torch.sort(perm, dim=1).values, expected)


def build_superpixel_graph_token_permutation(
    feat_nchw,
    soft_slic,
    k_spatial=3,
    k_feature=3,
    alpha=0.5,
    beta=0.5,
    token_inner_order='raster',
    detach_order=True,
):
    """Build a dense-token permutation from a SoftSLIC region graph path.

    Args:
        feat_nchw: Tensor[B, C, H, W].
        soft_slic: SoftSLIC module used to compute soft assignments.

    Returns:
        perm: Tensor[B, L], graph-ordered dense token indices.
        inv_perm: Tensor[B, L], inverse permutation back to raster order.
        stats: dict of detached assignment/order diagnostics.
    """
    if token_inner_order != 'raster':
        raise ValueError("token_inner_order currently supports only 'raster'.")

    b, _, h, w = feat_nchw.shape
    l = h * w
    device = feat_nchw.device

    def _build():
        slic_out = soft_slic(feat_nchw)
        if len(slic_out) == 5:
            region_tokens, Q, _, region_xy, slic_stats = slic_out
        else:
            region_tokens, Q, _, region_xy = slic_out
            slic_stats = {}

        # labels: [B, L], each dense token assigned to its strongest region.
        labels = Q.argmax(dim=-1)
        A = build_region_graph(
            region_tokens,
            region_xy,
            k_spatial=k_spatial,
            k_feature=k_feature,
            alpha=alpha,
            beta=beta,
        )
        region_order = greedy_graph_path(A, region_xy)
        region_rank = invert_permutation(region_order)

        # token_region_rank: [B, L], region order rank for every dense token.
        token_region_rank = torch.gather(region_rank, dim=1, index=labels)
        local_order = torch.arange(l, device=device).unsqueeze(0).expand(b, -1)
        score = token_region_rank * l + local_order

        perm = torch.argsort(score, dim=1, stable=True)
        inv_perm = invert_permutation(perm)
        stats = _compute_assignment_stats(
            Q,
            feat_h=h,
            feat_w=w,
            slic_stats=slic_stats,
            eps=getattr(soft_slic, 'eps', 1e-6),
        )
        stats['perm_valid'] = _permutation_is_valid(perm)
        return perm, inv_perm, stats

    if detach_order:
        with torch.no_grad():
            perm, inv_perm, stats = _build()
    else:
        perm, inv_perm, stats = _build()
        perm = perm.detach()
        inv_perm = inv_perm.detach()

    return perm.long(), inv_perm.long(), stats


__all__ = ['build_superpixel_graph_token_permutation', 'batched_gather_tokens']
