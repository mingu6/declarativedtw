import numpy as np

from numba import jit, prange
import torch
from torch.autograd import Function

from tslearn.metrics import dtw_subsequence_path as tsl_dtw_path_ss
from tslearn.metrics import dtw_path as tsl_dtw_path

import dilate_loss


class DTWLayer(torch.nn.Module):
    def __init__(self, subseq=True):
        super().__init__()
        self.dtw_path = tsl_dtw_path_ss if subseq else tsl_dtw_path

    def forward(self, x, y, x_t, y_t):
        B, N, _ = x.shape
        dtw_cost = torch.empty(B, dtype=x.dtype, device=x.device)
        x_np, y_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()
        w_ts, w_vs = [], []
        for b in range(B):
            dtw_path, cost = self.dtw_path(x_np[b, ...], y_np[b, ...])
            x_inds, y_inds = zip(*dtw_path)
            dtw_cost[b] = ((x[b, x_inds, :] - y[b, y_inds, :]) ** 2.).sum()
            x_inds_rc, y_inds_rc = zip(*[(s, t) for s, t in dict(dtw_path).items()])  # right continuous representation of path
            w_t, w_v = x_t[b, x_inds_rc], y_t[b, y_inds_rc]  # convert to warp times from indices
            w_ts.append(w_t)
            w_vs.append(w_v)
        return dtw_cost, torch.stack(w_ts), torch.stack(w_vs)


class DILATE(torch.nn.Module):
    def __init__(self, alpha=0.0, gamma=1.0, subseq=True):
        super().__init__()
        self.dtw = DTWLayer(subseq=subseq)
        self.alpha = alpha
        self.gamma = gamma
        self.subseq = subseq

    def forward(self, x, y, gt_x, gt_y):
        # compute DILATE loss
        Omega = dilate_loss.build_Omega(gt_x, gt_y, subseq=self.subseq)
        loss, _, _ = dilate_loss.dilate_loss(x, y, Omega, self.alpha, self.gamma, x.device, subseq=self.subseq)
        return loss


class SoftDTW(torch.nn.Module):
    def __init__(self, gamma=1.0, normalize=False, subseq=False):
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.subseq = subseq
        self.func_dtw = _SoftDTW.apply

    @staticmethod
    def calc_distance_matrix(x, y, subseq=False):
        b = x.size(0)
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        dist = torch.pow(x - y, 2).sum(3)
        if subseq:
            z = torch.zeros((b, n, 1), device=dist.device, dtype=dist.dtype)
            dist = torch.concat((z, dist, torch.zeros_like(z)), dim=2)
        return dist

    def forward(self, x, y):
        assert len(x.shape) == len(y.shape)
        squeeze = False
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            squeeze = True
        if self.normalize:  # not working for subsequences yet
            D_xy = self.calc_distance_matrix(x, y)
            out_xy = self.func_dtw(D_xy, self.gamma)
            D_xx = self.calc_distance_matrix(x, x)
            out_xx = self.func_dtw(D_xx, self.gamma)
            D_yy = self.calc_distance_matrix(y, y)
            out_yy = self.func_dtw(D_yy, self.gamma)
            result = out_xy - 1 / 2 * (out_xx + out_yy)  # distance
        else:
            D_xy = self.calc_distance_matrix(x, y, subseq=self.subseq)
            out_xy = self.func_dtw(D_xy, self.gamma)
            result = out_xy  # discrepancy
        return result.squeeze(0) if squeeze else result


@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for k in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                r0 = -R[k, i - 1, j - 1] / gamma
                r1 = -R[k, i - 1, j] / gamma
                r2 = -R[k, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[k, i, j] = D[k, i - 1, j - 1] + softmin
    return R


@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]


class _SoftDTW(Function):
    @staticmethod
    def forward(ctx, D, gamma):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        R = torch.Tensor(compute_softdtw(D_, g_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None
