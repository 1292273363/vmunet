import torch


def argsort_xy(region_xy, mode='yx'):
    """Return a batched lexicographic order over region coordinates."""
    if mode not in {'yx', 'xy'}:
        raise ValueError("mode must be either 'yx' or 'xy'")

    primary_idx, secondary_idx = (1, 0) if mode == 'yx' else (0, 1)
    # Stable secondary sort followed by stable primary sort gives batched
    # lexicographic ordering without relying on hand-tuned scalar keys.
    secondary_perm = torch.argsort(region_xy[..., secondary_idx], dim=1, stable=True)
    primary_values = torch.gather(region_xy[..., primary_idx], dim=1, index=secondary_perm)
    primary_perm = torch.argsort(primary_values, dim=1, stable=True)
    return torch.gather(secondary_perm, dim=1, index=primary_perm)


def invert_permutation(perm):
    """Invert batched permutations.

    Args:
        perm: Tensor[B, K]

    Returns:
        inverse: Tensor[B, K]
    """
    inverse = torch.empty_like(perm)
    base = torch.arange(perm.size(1), device=perm.device).unsqueeze(0).expand_as(perm)
    inverse.scatter_(1, perm, base)
    return inverse


def batched_gather_tokens(x, perm):
    """Gather batched token sequences with a batched permutation.

    Args:
        x: Tensor[B, K, ...]
        perm: Tensor[B, K]

    Returns:
        gathered: Tensor[B, K, ...]
    """
    index = perm.view(perm.size(0), perm.size(1), *([1] * (x.dim() - 2)))
    index = index.expand(-1, -1, *x.shape[2:])
    return torch.gather(x, dim=1, index=index)


def greedy_graph_path(A, region_xy):
    """Construct a batched greedy traversal path over the region graph."""
    with torch.no_grad():
        b, k, _ = A.shape
        device = A.device
        batch_idx = torch.arange(b, device=device)

        path = torch.empty(b, k, dtype=torch.long, device=device)
        visited = torch.zeros(b, k, dtype=torch.bool, device=device)

        current = argsort_xy(region_xy, mode='yx')[:, 0]
        path[:, 0] = current
        visited[batch_idx, current] = True

        for step in range(1, k):
            scores = A[batch_idx, current].masked_fill(visited, float('-inf'))
            best_score, next_idx = scores.max(dim=1)

            # If the sparse graph offers no positive continuation, fall back to
            # the nearest unvisited region in coordinate space.
            disconnected = best_score <= 0
            if disconnected.any():
                current_xy = region_xy[batch_idx, current].unsqueeze(1)
                spatial_dist = torch.norm(region_xy - current_xy, dim=-1)
                spatial_dist = spatial_dist.masked_fill(visited, float('inf'))
                fallback_idx = spatial_dist.argmin(dim=1)
                next_idx = torch.where(disconnected, fallback_idx, next_idx)

            path[:, step] = next_idx
            visited[batch_idx, next_idx] = True
            current = next_idx

        return path
