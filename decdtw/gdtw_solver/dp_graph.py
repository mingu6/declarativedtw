from typing import Tuple

import torch
from torch import Tensor

from decdtw.utils import BatchedSignal
import decdtw.gdtw_solver.utils as dutils


def feature_node_cost_l2_alt(signal1: BatchedSignal, signal2: BatchedSignal, warp_fn_times: Tensor,
                             warp_fn_discr_vals: Tensor) -> Tensor:
    """
    Computes feature node costs between two signals for dynamic programming graph. Assumes linear interpolation
    for features between observed points in both signals and also assumes feature cost is using l2 loss.

    Method:
    key observations are that for vectors z1, z2 that ||z1 - z2||^2 = z1^T z_1 - 2 z1^T z2 + z2^T z2, i.e.
    that we can write the squared norm as a linear sum of inner products of the vectors. Assuming z1 is on the line
    segment interpolating vectors x1, x2, i.e. z1 = (1 - a) * x1 + a * x2 for a in [0, 1], and z2 is on the line
    segment interpolating vectors y1, y2, i.e. z2 = (1 - b) * y1 + b * y2, we can use the bilinearity of the inner
    product to write ||z1 - z2||^2 as a linear combination of inner products x1^T x1, x2^T x2, x1^T x2, x1^T y1,
    x1^T y2, x2^T y1, x2^T y2, y1^T y2, y2^Ty2.

    This method yields lower memory cost compared to the feature_node_cost function since that function
    evaluates signal1 at all warp values. For long time series with high dimensional embeddings this
    will have a very high memory cost, especially for a GPU.

    :param signal1: reference signal where warp function is applied
    :param signal2: query signal with no warp applied
    :param warp_fn_times: (BxN) warp function times where query signal is evaluated
    :param warp_fn_discr_vals: (BxNxM) warp function value discretisation where reference signal is evaluated
    :return: returns feature node costs (BxNxM) for all timesteps
    """
    B, N, M = warp_fn_discr_vals.shape
    warp_discr_vals_flat = warp_fn_discr_vals.flatten(start_dim=1)
    # indices for upper/lower boundaries of observed times assoc. with desired eval times
    s1_ub_ind = torch.searchsorted(signal1.times, warp_discr_vals_flat, right=False)
    s1_lb_ind = torch.clamp(s1_ub_ind - 1, min=0)
    s2_ub_ind = torch.searchsorted(signal2.times, warp_fn_times, right=False)
    s2_lb_ind = torch.clamp(s2_ub_ind - 1, min=0)
    # interpolation proportion b/w upper/lower bounds for desired eval times in [0, 1]
    a = torch.gather(signal1.times, dim=1, index=s1_lb_ind)
    a = (warp_discr_vals_flat - a) / (torch.gather(signal1.times, dim=1, index=s1_ub_ind) - a)
    torch.nan_to_num(a, posinf=0., out=a)
    b = torch.gather(signal2.times, dim=1, index=s2_lb_ind)
    b = (warp_fn_times - b) / (torch.gather(signal2.times, dim=1, index=s2_ub_ind) - b)
    torch.nan_to_num(b, posinf=0., out=b)
    b = b.unsqueeze(2).expand(-1, -1, M).flatten(start_dim=1)
    a_c = 1. - a
    b_c = 1. - b
    # compute required inner products to compute squared norm
    xTy = signal1.values @ signal2.values.transpose(1, 2)
    xTx = torch.einsum('bnm,bnm->bn', signal1.values, signal1.values)
    yTy = torch.einsum('bnm,bnm->bn', signal2.values, signal2.values)
    x1Tx2 = torch.einsum('bnm,bnm->bn', signal1.values[:, :-1, :], signal1.values[:, 1:, :])
    y1Ty2 = torch.einsum('bnm,bnm->bn', signal2.values[:, :-1, :], signal2.values[:, 1:, :])
    # gather inner products at required points
    batch_inds = torch.arange(B).unsqueeze(1).expand(-1, N * M)
    dutils.adjust_node_costs_trapezoidal_inplace(l2_costs, warp_fn_times)
    return l2_costs


def feature_node_cost(signal1: BatchedSignal, signal2: BatchedSignal, warp_fn_times: Tensor,
                      warp_fn_discr_vals: Tensor, p: float = 2) -> Tensor:
    """
    Computes feature loss between two signals at all given evaluation times.

    :param signal1: reference signal where warp function is applied
    :param signal2: query signal with no warp applied
    :param warp_fn_times: (BxN) warp function times where query signal is evaluated
    :param warp_fn_discr_vals: (BxNxM) warp function value discretisation where reference signal is evaluated
    :param p: degree of Lp loss to use (e.g. squared Euclidean p=2)
    :return: returns feature node costs (BxNxM) for all timesteps
    """
    # if discretised values are at the observed times, don't bother interpolating! saves compute/mem
    if warp_fn_times.shape[1] == signal2.times.shape[1] and torch.allclose(warp_fn_times, signal2.times):
        s2_feats = signal2.values
    else:
        s2_feats = signal2(warp_fn_times)
    if warp_fn_discr_vals.shape[2] == signal1.times.shape[1] and torch.allclose(warp_fn_discr_vals, signal1.times.unsqueeze(1)):
        s1_feats = signal1.values.unsqueeze(1)
    else:
        s1_feats = signal1(warp_fn_discr_vals)
    costs = torch.cdist(s1_feats, s2_feats.unsqueeze(2), p=p).squeeze(-1) ** p
    dutils.adjust_node_costs_trapezoidal_inplace(costs, warp_fn_times)
    return costs


def reg_edge_cost(warp_fn_times: Tensor, warp_fn_discr_vals: Tensor, grad_lb: Tensor, grad_ub: Tensor,
                  warp_deriv_mean: Tensor, reg_wt: Tensor, p: float, increm_edge: bool = False) -> Tensor:
    """
    Computes regularisation cost which are used as edge costs for identifying the minimal cost path.
    Costs are inf where gradient bounds are violated. Costs are computed in a single vectorised step rather than
    computed online per timestep, saving computation time in exchange for memory usage.

    :param warp_fn_times: (BxN) warp times associated to discretised warp function values
    :param warp_fn_discr_vals: (BxNxM) discrete warp function values from which the instantaneous warp is determined
    :param grad_lb: (BxN-1) lower bound on warp gradient where elem i is for t=i to i+1
    :param grad_ub: (BxN-1) upper bound on warp gradient where elem i is for t=i to i+1
    :param warp_deriv_mean: (B,) per-element expected warp derivative to apply penalty from (e.g. 1)
    :param reg_wt: (B,) with per-element weighting applied to instantaneous reg. cost in objective function
    :param p: degree of Lp loss to use on warp derivative
    :param increm_edge: generates edge weights timestep by timestep rather than in one hit. Saves GPU memory at the
                        cost of inference time. Especially useful if getting OOM errors for long time series.
    :return: returns Bx(N-1)xMxM set of warp regularisation costs for all timesteps where the t, i, j coordinate
             in the final (N-1)xMxM block represents the cost of transitioning to the ith warp function value at time t+1
             from the jth warp function value at t.
    """
    B, N, M = warp_fn_discr_vals.shape
    dtime = torch.diff(warp_fn_times, dim=1)[..., None, None]
    if increm_edge:
        for t in range(1, N):
            dwarp = warp_fn_discr_vals[:, t, :, None] - warp_fn_discr_vals[:, t-1, None, :]
            warp_deriv = dwarp / dtime[:, t-1, ...]
            reg_cost = reg_wt[..., None] * (warp_deriv - warp_deriv_mean[..., None]) ** p * dtime[:, t-1, ...]
            # violation of constraints have infinite cost
            violate_mask = torch.logical_or(warp_deriv < grad_lb[:, t-1, None, None],
                                            warp_deriv > grad_ub[:, t-1, None, None])
            dutils.index_and_set_inf(reg_cost, violate_mask)
            yield reg_cost[:, ...]
    else:
        dwarp = warp_fn_discr_vals[:, 1:, :, None] - warp_fn_discr_vals[:, :-1, None, :]
        warp_deriv = dwarp / dtime
        reg_cost = reg_wt[:, None, None] * (warp_deriv - warp_deriv_mean[:, None, None]) ** p * dtime
        # violation of constraints have infinite cost
        violate_mask = torch.logical_or(warp_deriv < grad_lb[..., None, None], warp_deriv > grad_ub[..., None, None])
        dutils.index_and_set_inf(reg_cost, violate_mask)
        for t in range(N-1):
            yield reg_cost[:, t, ...]

def warp_fn_glb_bounds(warp_fn_times: Tensor, signal1_times: Tensor, local_lb_vals: Tensor, local_ub_vals: Tensor,
                       glb_band_lb: Tensor, glb_band_ub: Tensor, subseq_enabled: bool) -> Tuple[Tensor, Tensor]:
    """
    Evaluates time-dependent global bounds on the time warp function at provided evaluation times. Bounds are either
    explicit (e.g. RK/SC band) or implicitly deduced based on warp gradients. Note assumes local constraints have
    already been rescaled by the average warp derivative.

    :param warp_fn_times: (BxN) warp function time values to evaluate bounds at
    :param signal1_times: (BxN) reference signal observed times to which the warp function is applied
    :param local_lb_vals: (BxN-1) time-varying warp gradient lower bounds where elem i relates to gradient from t=i to i+1
    :param local_ub_vals: (BxN-1) time-varying warp gradient upper bounds where elem i relates to gradient from t=i to i+1
    :param glb_band_lb: (BxN) time-varying global lower bound relating constraints (e.g. R-K band)
    :param glb_band_ub: (BxN) time-varying global upper bound relating constraints (e.g. R-K band)
    :param subseq_enabled: bounds account for subsequence alignment
    :return: (BxN) tensors containing warp function upper and lower bounds at each given timestep
    """
    B, N = warp_fn_times.shape
    assert local_lb_vals.shape == (B, N-1), 'local gradient constraints not consistent with warp time sizes'
    assert local_lb_vals.shape == local_ub_vals.shape, 'lower and upper local gradient constraints not consistent'
    assert glb_band_lb.shape == warp_fn_times.shape, 'lower global constraints not consistent with warp time sizes'
    assert glb_band_ub.shape == warp_fn_times.shape, 'upper global constraints not consistent with warp time sizes'
    # global warp bounds set by gradient constraints
    dt = torch.diff(warp_fn_times, dim=1)
    warp_fn_max_values = signal1_times.max(dim=1, keepdim=True).values
    if not subseq_enabled:
        assert torch.all(torch.sum(local_ub_vals * dt, dim=1, keepdim=True) >= warp_fn_max_values), \
            'upper endpoint constraints are impossible to satisfy for sequence alignment problem due to upper bounds'
        assert torch.all(torch.sum(local_lb_vals * dt, dim=1, keepdim=True) <= warp_fn_max_values), \
            'upper endpoint constraints are impossible to satisfy for sequence alignment problem due to lower bounds'
    # Implements gradient bounds equation (7) from Deriso and Boyd paper for gradient bounds if no subseq align
    if subseq_enabled:
        lb_local = torch.zeros_like(warp_fn_times)
        ub_local = warp_fn_max_values.clone().expand(-1, warp_fn_times.shape[1])
    else:
        # upper/lower bounds can take be one of two options (see Deriso and Boyd paper)
        warp_ub0 = warp_fn_max_values - torch.flip(torch.cumsum(local_lb_vals * dt, dim=1), [1])
        warp_ub0 = torch.concat((warp_ub0, warp_fn_max_values), dim=1)
        warp_lb0 = torch.cumsum(local_lb_vals * dt, dim=1)
        warp_lb0 = torch.concat((torch.zeros_like(warp_fn_max_values), warp_lb0), dim=1)

        warp_ub1 = torch.cumsum(local_ub_vals * dt, dim=1)
        warp_ub1 = torch.concat((torch.zeros_like(warp_fn_max_values), warp_ub1), dim=1)
        warp_lb1 = warp_fn_max_values - torch.flip(torch.cumsum(local_ub_vals * dt, dim=1), [1])
        warp_lb1 = torch.concat((warp_lb1, warp_fn_max_values), dim=1)

        ub_local = torch.minimum(warp_ub0, warp_ub1)
        lb_local = torch.maximum(warp_lb0, warp_lb1)
    # update time-varying global band constraints (RK-band)
    warp_lb = torch.maximum(glb_band_lb, lb_local)
    warp_ub = torch.minimum(glb_band_ub, ub_local)
    # endpoint constraints if no subsequence alignment
    if not subseq_enabled:
        warp_lb[:, -1] = warp_fn_max_values.squeeze()  # this line enforces end alignment constraint for warp fn
        warp_ub[:, 0] = 0.  # this line enforces start alignment constraint for warp fn
    # ensure bounds are within time series durations (absolute bounds of valid times)
    warp_lb = torch.clamp(warp_lb, torch.zeros_like(warp_lb), warp_fn_max_values)
    warp_ub = torch.clamp(warp_ub, torch.zeros_like(warp_ub), warp_fn_max_values)
    return warp_lb, warp_ub
