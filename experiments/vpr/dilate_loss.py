# taken directly from https://github.com/vincent-leguen/DILATE w/modifications for temporal loss

import torch
import soft_dtw
import path_soft_dtw


def build_Omega(gt_q, gt_r, subseq=False):
    """
    Construct DILATE temporal loss Omega from ground truth warps

    :param gt_q: B x N_q x 2 query image GPS
    :param gt_r: B x N_r x 2 reference image GPS
    :return: B x N_q x N_r temporal loss matrix
    """
    B, N_q, _ = gt_q.shape
    _, N_r, _ = gt_r.shape
    Omega = torch.linalg.norm(gt_q.unsqueeze(2).expand(-1, N_q, N_r, 2) - gt_r.unsqueeze(1).expand(-1, N_q, N_r, 2), dim=3)
    if subseq:
        z = torch.zeros((B, N_q, 1), dtype=gt_q.dtype, device=gt_q.device)
        Omega = torch.cat((z, Omega, z), dim=2)
    return Omega


def dilate_loss(outputs, targets, Omega, alpha, gamma, device, subseq=False):
    # outputs: shape (B, N_o, d)
    # targets: shape (B, N_t, d)
    B, N_o, d = outputs.shape
    _, N_t, d = targets.shape
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    D = torch.zeros((B, N_o, N_t)).to(device)
    for k in range(B):
        Dk = soft_dtw.pairwise_distances(outputs[k, :, :], targets[k, :, :])
        D[k:k + 1, :, :] = Dk
    if subseq:
        z = torch.zeros((B, N_o, 1), device=outputs.device, dtype=outputs.dtype)
        D = torch.concat((z, D, z), dim=2)
    loss_shape = softdtw_batch(D, gamma)

    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)
    loss_temporal = torch.sum(path * Omega, dim=[1, 2]) / (N_o * N_t)
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal
