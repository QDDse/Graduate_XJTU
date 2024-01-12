import torch
import torch.nn.functional as F
import torch.nn as nn


def PA_loss(patches_visible, patches_ir, temperature=0.1, logit_scale=None):
    """
    InfoNCE loss with cosine similarity for visible and infrared image patches.

    Args:
    - patches_visible (torch.Tensor): Tensor of shape [B, N, D] containing patch embeddings from the visible image.
    - patches_ir (torch.Tensor): Tensor of shape [B, N, D] containing patch embeddings from the infrared image.
    - temperature (float): Temperature scaling factor for the InfoNCE loss.

    Returns:
    - torch.Tensor: Scalar tensor containing the InfoNCE loss.
    """
    loss_pa = 0.0
    B = patches_ir.shape[0]
    for b in range(B):
        # Normalize patch embeddings to unit vectors
        vis_patch = F.normalize(patches_visible[b], p=2, dim=-1)
        ir_patch = F.normalize(patches_ir[b], p=2, dim=-1)

        # Compute cosine similarity between all pairs of patches
        # Shape [B, N, N], where N is the number of patches
        logits_per_vis = (
            torch.matmul(vis_patch, ir_patch.transpose(-2, -1)) / temperature
        )
        logits_per_ir = (
            torch.matmul(ir_patch, vis_patch.transpose(-2, -1)) / temperature
        )
        if logit_scale is not None:
            logits_per_vis *= logits_per_vis
            logits_per_ir *= logits_per_ir

        # Positive samples are on the diagonal of the similarity matrix
        # Use a mask to select them
        labels = torch.arange(logits_per_vis.shape[0]).to(
            logits_per_vis.device
        )  # [144]；
        vis_ir_loss = (
            F.cross_entropy(logits_per_vis, labels)
            + F.cross_entropy(logits_per_ir, labels) / 2
        )

        loss_pa += vis_ir_loss

    return loss_pa / B
