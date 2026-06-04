import torch
import torch.nn.functional as F


def build_region_graph(region_tokens, region_xy, k_spatial=6, k_feature=6, alpha=0.5, beta=0.5):
    """Build a symmetric sparse affinity graph for region tokens.

    Args:
        region_tokens: Tensor[B, K, C]
        region_xy: Tensor[B, K, 2]

    Returns:
        A: Tensor[B, K, K]
    """
    _, k, _ = region_tokens.shape
    device = region_tokens.device

    # Dense affinities before top-k sparsification: [B, K, K].
    spatial_affinity = torch.exp(-torch.cdist(region_xy, region_xy))
    normalized_tokens = F.normalize(region_tokens, dim=-1, eps=1e-6)
    feature_affinity = torch.bmm(normalized_tokens, normalized_tokens.transpose(1, 2))

    eye = torch.eye(k, device=device, dtype=torch.bool).unsqueeze(0)
    spatial_scores = spatial_affinity.masked_fill(eye, float('-inf'))
    feature_scores = feature_affinity.masked_fill(eye, float('-inf'))

    spatial_k = min(k_spatial, max(k - 1, 0))
    feature_k = min(k_feature, max(k - 1, 0))

    A_spatial = torch.zeros_like(spatial_affinity)
    A_feature = torch.zeros_like(feature_affinity)

    if spatial_k > 0:
        spatial_values, spatial_idx = torch.topk(spatial_scores, k=spatial_k, dim=-1)
        A_spatial.scatter_(-1, spatial_idx, spatial_values)

    if feature_k > 0:
        feature_values, feature_idx = torch.topk(feature_scores, k=feature_k, dim=-1)
        A_feature.scatter_(-1, feature_idx, feature_values)

    # A: [B, K, K], no self-loop and symmetric for downstream path construction.
    A = alpha * A_spatial + beta * A_feature
    A = A.masked_fill(eye, 0.0)
    A = 0.5 * (A + A.transpose(1, 2))
    return A
