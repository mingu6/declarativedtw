from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch import Tensor
from scipy import sparse as sparse, optimize as sopt
from torch.autograd import grad

from decdtw.utils import BatchedSignal, dtw_objective


def linear_dtw_constraints(times: np.ndarray, grad_min: np.array, grad_max: np.array):
    dt = np.diff(times)
    n = times.shape[0]
    A = sparse.lil_matrix((n - 1, n), dtype=times.dtype)
    A.setdiag(-1., k=0)
    A.setdiag(1., k=1)
    lb = dt * grad_min
    ub = dt * grad_max
    return [sopt.LinearConstraint(A.tocsr(), lb, ub)]


def sp_nlp_dtw_bounds(n: int, warp_max: float, band_lb: np.array, band_ub: np.array, subseq: bool, dtype=np.float32):
    val_lb = np.zeros(n, dtype=dtype)
    val_ub = np.full_like(val_lb, warp_max)
    val_lb = np.maximum(band_lb, val_lb)
    val_ub = np.minimum(band_ub, val_ub)
    if not subseq:
        val_lb[-1] = warp_max
        val_ub[0] = 0.
    return sopt.Bounds(val_lb, val_ub)


def refine_soln_nlp(warp_fn: BatchedSignal, signal1: BatchedSignal, signal2: BatchedSignal, band_lb: Tensor, band_ub: Tensor,
                    grad_min: Tensor, grad_max: Tensor, reg_wt: Tensor, exp_warp_grad: Tensor,
                    f_p: float, r_p: float, subseq: bool) -> Tuple[Tensor, Tensor]:
    dev = warp_fn.values.device
    with torch.enable_grad():
        warp_max = signal1.times.max(dim=1).values
        # objective w.r.t. warp only for solver
        def f(warp, idx):
            w = BatchedSignal(torch.from_numpy(warp).unsqueeze(0).to(warp_fn.values.dtype).to(dev),
                              times=warp_fn.times[idx].unsqueeze(0))
            obj = dtw_objective(signal1[idx], signal2[idx], w, reg_wt[idx], exp_warp_grad[idx], f_p, r_p).squeeze(0)
            return obj.cpu().item()

        # jacobian of above objective -> allows speedup from computing numerically using solver
        def df(warp, idx):
            warp_tc = torch.from_numpy(warp).to(warp_fn.values.dtype).to(dev)
            warp_tc.requires_grad = True
            w = BatchedSignal(warp_tc.unsqueeze(0), times=warp_fn.times[idx].unsqueeze(0))
            obj = dtw_objective(signal1[idx], signal2[idx], w, reg_wt[idx], exp_warp_grad[idx], f_p, r_p).squeeze(0)
            dobj = grad(obj, warp_tc)[0]
            return dobj.cpu().numpy()

        B, N = warp_fn.values.shape
        refined_warp_vals = torch.empty_like(warp_fn.values)
        costs = torch.empty((B,), dtype=reg_wt.dtype, device=reg_wt.device)
        for b in range(B):
            warp_times = warp_fn.times[b].detach().cpu().numpy()
            warp_vals = warp_fn.values[b].detach().cpu().numpy()
            exp_warp_grad_ = exp_warp_grad[b].detach().cpu().numpy()
            dtype = warp_vals.dtype
            glb_lb = band_lb[b].detach().cpu().numpy()
            glb_ub = band_ub[b].detach().cpu().numpy()
            gmin = grad_min[b].detach().cpu().numpy()
            gmax = grad_max[b].detach().cpu().numpy()
            constraints = linear_dtw_constraints(warp_times, gmin, gmax)
            bounds = sp_nlp_dtw_bounds(N, warp_max[b].item(), glb_lb, glb_ub, subseq, dtype)
            try:
                res = sopt.minimize(f, warp_vals, method='SLSQP', jac=df, args=(b,), bounds=bounds, constraints=constraints)
            except:
                # try numerical if autograd fails
                res = sopt.minimize(f, warp_vals, method='SLSQP', args=(b,), bounds=bounds, constraints=constraints)
            if not res.success:
                raise RuntimeError(f"SLSQP failed to find a solution during NLP refinement for element {b}")
            refined_warp_vals[b, :] = torch.from_numpy(res.x).to(warp_fn.values.device)
            costs[b] = res.fun
    return costs, refined_warp_vals