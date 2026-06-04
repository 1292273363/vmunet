import torch
import torch.nn.functional as F


def _resize_target_to_feat(target, feat_hw):
    if target.dim() == 3:
        target = target.unsqueeze(1)
    return F.interpolate(target.float(), size=feat_hw, mode='nearest').squeeze(1)


def region_label_reconstruction_loss(Q, target, feat_hw, num_classes, ignore_index=None, eps=1e-6):
    """Reconstruct task labels through the soft region assignment matrix.

    Args:
        Q: Tensor[B, N, K]
        target: Tensor[B, H, W] or Tensor[B, 1, H, W]
        feat_hw: tuple[int, int]
        num_classes: int
        ignore_index: optional int
    """
    b, n, _ = Q.shape
    class_count = max(num_classes, 2) if num_classes == 1 else num_classes

    # labels: [B, N]
    labels = _resize_target_to_feat(target, feat_hw)
    if num_classes == 1:
        labels = (labels > 0.5).long()
    else:
        labels = labels.long()
    labels = labels.reshape(b, n)

    if ignore_index is None:
        valid = torch.ones_like(labels, dtype=Q.dtype)
        safe_labels = labels
    else:
        valid = (labels != ignore_index).to(Q.dtype)
        safe_labels = labels.masked_fill(labels == ignore_index, 0)

    # target_one_hot: [B, N, num_classes]
    target_one_hot = F.one_hot(safe_labels.clamp_min(0), num_classes=class_count).to(Q.dtype)
    target_one_hot = target_one_hot * valid.unsqueeze(-1)

    # region_label: [B, K, num_classes]
    weighted_Q = Q * valid.unsqueeze(-1)
    region_mass = weighted_Q.sum(dim=1).unsqueeze(-1).clamp_min(eps)
    region_label = torch.bmm(weighted_Q.transpose(1, 2), target_one_hot) / region_mass

    # pred: [B, N, num_classes]
    pred = torch.bmm(Q, region_label).clamp_min(eps)
    ce = -(target_one_hot * pred.log()).sum(dim=-1)
    return (ce * valid).sum() / valid.sum().clamp_min(1.0)


def region_compactness_loss(Q, region_xy, feat_hw):
    """Penalize soft assignments whose reconstructed coordinates drift away."""
    b, n, _ = Q.shape
    h, w = feat_hw

    ys = torch.linspace(0.0, 1.0, h, device=Q.device, dtype=Q.dtype)
    xs = torch.linspace(0.0, 1.0, w, device=Q.device, dtype=Q.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    # xy_grid: [B, N, 2], xy_rec: [B, N, 2]
    xy_grid = torch.stack([xx, yy], dim=-1).reshape(1, n, 2).expand(b, -1, -1)
    xy_rec = torch.bmm(Q, region_xy)
    return ((xy_grid - xy_rec) ** 2).sum(dim=-1).mean()


def region_mass_balance_loss(Q, eps=1e-6):
    """Encourage each region to receive a balanced amount of assignments.

    Args:
        Q: Tensor[B, N, K]
    """
    _, n, k = Q.shape
    mass = Q.sum(dim=1)  # [B, K]
    target = float(n) / float(k)
    mass_norm = mass / max(target, eps)
    return ((mass_norm - 1.0) ** 2).mean()
