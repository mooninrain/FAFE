import torch

from typing import TypeAlias

Tensor: TypeAlias = torch.Tensor

def frame_aligned_point_error_loss(
    R_pred: Tensor,
    t_pred: Tensor,
    R_gt: Tensor,
    t_gt: Tensor,
    frame_mask: Tensor,
    pos_pred: Tensor,
    pos_gt: Tensor,
    pos_mask: Tensor,
    axis_scale: float = 10.0,
    dist_clamp: float | None = None,
    pair_mask: Tensor | None = None,
    eps: float = 1e-4,
    group_aware: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute frame aligned point error.

    See alphafold supplementary 1.9.2 for details.

    Args:
        R_pred: Predicted rotation matrices of frames, (B, N, 3, 3).
        t_pred: Predicted translations of frames, (B, N, 3).
        R_gt: Ground truth rotation matrices of frames, (B, N, 3, 3).
        t_gt: Ground truth translations of frames, (B, N, 3).
        frame_mask: Existing masks of ground truth frames, (B, N).
        pos_pred: Predicted positions of points, (B, M, 3).
        pos_gt: Ground truth positions of points, (B, M, 3).
        pos_mask: Existing masks of ground truth points, (B, M).
        axis_scale: Scale by which the loss is divided. Usually 10 for nanometers.
        dist_clamp: Cutoff above which distance errors are disregarded.
        pair_mask: Additional pair masks of pairs which should be calculated, (B, N, M) or None.
            pair_mask=True, the FAPE loss is calculated; vice not calculated.
            If None, all pairs are calculated.
        eps: Small number for numeric safety.

        For example:
        +------------------+------------------+-----------------+-----------------+
        |       dist       |    dist_clamp    |    pair_mask    |      loss       |
        +------------------+------------------+-----------------+-----------------+
        |        5 A       |       10 A       |       True      |       5 A       |
        +------------------+------------------+-----------------+-----------------+
        |       15 A       |       10 A       |       True      |      10 A       |
        +------------------+------------------+-----------------+-----------------+
        |       15 A       |       10 A       |       False     |       0 A       |
        +------------------+------------------+-----------------+-----------------+

    Returns:
        Fape loss, (B).
        Fape Error, (B, N, M).
    """
    if dist_clamp is None:
        dist_clamp = 1e9

    n_frames = t_pred.shape[1]
    pos_pred_local = common.geometric.global_to_local(
        R_pred, t_pred, einops.repeat(pos_pred, "... n_atoms t -> ... n_frames n_atoms t", n_frames=n_frames)
    )
    pos_gt_local = common.geometric.global_to_local(
        R_gt, t_gt, einops.repeat(pos_gt, "... n_atoms t -> ... n_frames n_atoms t", n_frames=n_frames)
    )

    dist_local = torch.sqrt(torch.sum((pos_pred_local - pos_gt_local) ** 2, dim=-1) + eps)

    frame_pos_cross_mask = tensor_utils.node2pair(frame_mask, pos_mask, -1, torch.logical_and)
    if pair_mask is not None:
        frame_pos_cross_mask = pair_mask * frame_pos_cross_mask

    dist_local = dist_local.clamp(max=dist_clamp) / axis_scale
    if not group_aware:
        loss, error = common.mask.masked_mean(dist_local, frame_pos_cross_mask, (-1, -2), return_masked=True)
    else:
        loss, error = common.mask.masked_quadratic_mean(dist_local, frame_pos_cross_mask, (-1, -2), return_masked=True)
    return loss, error.detach()