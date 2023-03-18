# taken directly from https://github.com/vincent-leguen/DILATE w/modifications for temporal loss

import torch
import soft_dtw
import path_soft_dtw


def build_Omega(gt_warps, s_t, p_t):
    """
    Construct DILATE temporal loss Omega from ground truth warps

    :param gt_warps: List (B) of tuple of N_gt,i, N_gt,i tensors (score gt, perf gt)
    :param s_t: B x N_s score observed times
    :param p_t: B x N_p perf observed times
    :return: B x N_s x N_p temporal loss matrix
    """
    B, N_s = s_t.shape
    _, N_p = p_t.shape
    Omega = torch.zeros((B, N_s, N_p), dtype=s_t.dtype, device=s_t.device)
    # for each gt point, find associated nearest score observation
    sc_inds = [torch.searchsorted(s_t[b], gt_warp[0]) for b, gt_warp in enumerate(gt_warps)]
    for b, sc_ind in enumerate(sc_inds):
        for t, ind in enumerate(sc_ind):
            Omega[b, ind, :] = torch.abs(gt_warps[b][1][t] - p_t[b])  # equivalent to timeerr
    return Omega


def dilate_loss(outputs, targets, Omega, alpha, gamma, device):
    # outputs: shape (B, N_o, d)
    # targets: shape (B, N_t, d)
    B, N_o, d = outputs.shape
    _, N_t, d = targets.shape
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    D = torch.zeros((B, N_o, N_t)).to(device)
    for k in range(B):
        Dk = soft_dtw.pairwise_distances(outputs[k, :, :], targets[k, :, :])
        D[k:k + 1, :, :] = Dk
    loss_shape = softdtw_batch(D, gamma)

    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)
    loss_temporal = torch.sum(path * Omega, dim=[1, 2]) / (N_o * N_t)
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal
