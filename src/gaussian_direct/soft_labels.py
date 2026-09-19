"""Soft residue and surface supervision targets."""

import torch as pt


def residue_centers_from_atom_mask(X, M):
    weights = M.to(dtype=X.dtype)
    denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
    return weights.transpose(0, 1).matmul(X) / denom


def residue_soft_labels(labels, mask, residue_centers, sigma=8.0, max_value=0.7, cutoff=24.0):
    """Spread hard positive residue labels to nearby residues with a Gaussian.

    Hard positives remain 1.0. Nearby hard negatives receive a value up to
    max_value, which makes uncertain interface-border residues less punitive
    without replacing the primary hard-label target.
    """
    labels = labels.float()
    mask = mask.float()
    if labels.numel() == 0:
        return labels
    valid = mask > 0.5
    positive = valid & (labels > 0.5)
    if not pt.any(positive):
        return labels.clone()
    centers = residue_centers.to(device=labels.device, dtype=pt.float)
    pos_centers = centers[positive]
    dist = pt.cdist(centers, pos_centers).min(dim=-1).values
    sigma = max(float(sigma), 1e-6)
    soft = pt.exp(-0.5 * (dist / sigma).square()) * float(max_value)
    if cutoff and float(cutoff) > 0:
        soft = pt.where(dist <= float(cutoff), soft, pt.zeros_like(soft))
    soft = pt.where(labels > 0.5, pt.ones_like(soft), soft)
    return pt.where(valid, soft.clamp(0.0, 1.0), labels)


def surface_soft_labels(
    labels,
    mask,
    residue_centers,
    surface_pos,
    surface_normal=None,
    sigma_tangent=6.0,
    sigma_normal=3.0,
    max_value=1.0,
    cutoff=24.0,
    mode="local_topk_weighted",
    topk=16,
):
    """Project hard positive residue labels onto surface points.

    The kernel separates tangent and normal components, matching S1's
    residue-to-surface projection geometry. The output is a soft target for
    surface logits, not a replacement for residue hard labels.
    """
    labels = labels.float()
    mask = mask.float()
    if labels.numel() == 0 or surface_pos.numel() == 0:
        return surface_pos.new_zeros((surface_pos.shape[0],), dtype=pt.float)
    valid = mask > 0.5
    positive = valid & (labels > 0.5)
    if not pt.any(positive):
        return surface_pos.new_zeros((surface_pos.shape[0],), dtype=pt.float)
    centers = residue_centers.to(device=surface_pos.device, dtype=surface_pos.dtype)
    if surface_normal is None:
        centroid = centers.mean(dim=0, keepdim=True)
        surface_normal = surface_pos - centroid
        surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normal_all = surface_normal.to(device=surface_pos.device, dtype=surface_pos.dtype)
    sigma_t = max(float(sigma_tangent), 1e-6)
    sigma_n = max(float(sigma_normal), 1e-6)
    mode = str(mode)
    if mode not in {"global_max", "local_topk_weighted", "positive_topk_noisy_or"}:
        raise ValueError("surface soft label mode must be one of: global_max, local_topk_weighted, positive_topk_noisy_or")

    if mode == "global_max":
        pos_centers = centers[positive]
        rel = pos_centers.unsqueeze(0) - surface_pos.unsqueeze(1)
        normal = normal_all.unsqueeze(1)
        d_normal = (rel * normal).sum(dim=-1).abs()
        d2 = (rel * rel).sum(dim=-1).clamp_min(0.0)
        d_tangent = (d2 - d_normal.square()).clamp_min(0.0).sqrt()
        score = -0.5 * ((d_tangent / sigma_t) ** 2 + (d_normal / sigma_n) ** 2)
        soft = pt.exp(score).max(dim=-1).values * float(max_value)
        if cutoff and float(cutoff) > 0:
            min_dist = d2.sqrt().min(dim=-1).values
            soft = pt.where(min_dist <= float(cutoff), soft, pt.zeros_like(soft))
        return soft.clamp(0.0, 1.0)

    if mode == "positive_topk_noisy_or":
        pos_centers = centers[positive]
        dist_pos = pt.cdist(surface_pos, pos_centers).clamp_min(0.0)
        k = min(max(int(topk), 1), int(pos_centers.shape[0]))
        local_dist, idx = pt.topk(dist_pos, k=k, dim=-1, largest=False)
        rel = pos_centers[idx] - surface_pos.unsqueeze(1)
        normal = normal_all.unsqueeze(1)
        d_normal = (rel * normal).sum(dim=-1).abs()
        d2 = (rel * rel).sum(dim=-1).clamp_min(0.0)
        d_tangent = (d2 - d_normal.square()).clamp_min(0.0).sqrt()
        score = -0.5 * ((d_tangent / sigma_t) ** 2 + (d_normal / sigma_n) ** 2)
        g = pt.exp(score).clamp(0.0, 1.0) * float(max_value)
        if cutoff and float(cutoff) > 0:
            g = pt.where(local_dist <= float(cutoff), g, pt.zeros_like(g))
        soft = 1.0 - (1.0 - g.clamp(0.0, 1.0)).prod(dim=-1)
        return soft.clamp(0.0, 1.0)

    dist = pt.cdist(surface_pos, centers).clamp_min(0.0)
    k = min(max(int(topk), 1), int(centers.shape[0]))
    local_dist, idx = pt.topk(dist, k=k, dim=-1, largest=False)
    rel = centers[idx] - surface_pos.unsqueeze(1)
    normal = normal_all.unsqueeze(1)
    d_normal = (rel * normal).sum(dim=-1).abs()
    d2 = (rel * rel).sum(dim=-1).clamp_min(0.0)
    d_tangent = (d2 - d_normal.square()).clamp_min(0.0).sqrt()
    score = -0.5 * ((d_tangent / sigma_t) ** 2 + (d_normal / sigma_n) ** 2)

    local_valid = valid.to(device=surface_pos.device)[idx]
    local_labels = labels.to(device=surface_pos.device, dtype=surface_pos.dtype)[idx]
    masked_score = score.masked_fill(~local_valid, -1e9)
    weight = pt.softmax(masked_score, dim=-1)
    weight = pt.where(local_valid, weight, pt.zeros_like(weight))
    denom = weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    weight = weight / denom
    soft = (weight * local_labels).sum(dim=-1) * float(max_value)
    if cutoff and float(cutoff) > 0:
        local_positive = local_valid & (local_labels > 0.5)
        inf = local_dist.new_full(local_dist.shape, float("inf"))
        min_pos_dist = pt.where(local_positive, local_dist, inf).min(dim=-1).values
        soft = pt.where(min_pos_dist <= float(cutoff), soft, pt.zeros_like(soft))
    return soft.clamp(0.0, 1.0)


def masked_soft_bce(logits, targets, mask=None, pos_weight=None):
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    if mask is None:
        valid = pt.ones_like(targets, dtype=pt.bool)
    else:
        valid = mask.to(device=logits.device) > 0.5
    weight = None
    if pos_weight is not None:
        pw = pos_weight.to(device=logits.device, dtype=logits.dtype)
        weight = 1.0 + (pw - 1.0) * targets
    element_loss = pt.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=weight, reduction="none"
    )
    valid_float = valid.to(dtype=element_loss.dtype)
    return (element_loss * valid_float).sum() / valid_float.sum().clamp_min(1.0)
