import torch

from typing import TypeAlias

Tensor: TypeAlias = torch.Tensor

def torsion_angle_loss(
    alpha: Tensor,
    torsion_gt: Tensor,
    torsion_mask: Tensor,
    torsion_alt: Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Compute torsion angle loss.

    See alphafold supplementary 1.9.1 for details.

    Args:
        alpha: Un-normalized sine and cosine of torsional angles, (B, N, 5, 2).
        torsion_gt: Ground truth torsional angles, (B, N, 5).
        torsion_mask: Ground truth masks of torsional angles, (B, N, 5).
        torsion_alt: Alternative ground truth torsional angles, (B, N, 5).
        eps: Small number for numeric safety.

    Returns:
        Torsion loss + 0.02 angle norm loss, (B).
        Torsion error + 0.02 angle norm error, (B, N, 5).
    """
    alpha_norm = torch.linalg.vector_norm(alpha, dim=-1)
    alpha = alpha / einops.rearrange(alpha_norm, "... -> ... 1").clamp(eps)
    alpha_gt = torch.stack([torsion_gt.sin(), torsion_gt.cos()], dim=-1)
    torsion_loss = torch.square(alpha - alpha_gt).sum(-1)

    if torsion_alt is not None:
        alpha_alt = torch.stack([torsion_alt.sin(), torsion_alt.cos()], dim=-1)
        torsion_loss_alt = torch.square(alpha - alpha_alt).sum(-1)
        torsion_loss = torch.minimum(torsion_loss_alt, torsion_loss)

    alpha_norm_loss = torch.abs(alpha_norm - 1)
    loss, error = common.mask.masked_mean(
        torsion_loss + alpha_norm_loss * 0.02, torsion_mask, (-1, -2), return_masked=True
    )

    return loss, error.detach()