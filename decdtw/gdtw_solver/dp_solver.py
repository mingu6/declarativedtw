import math
from typing import Tuple, Optional, Union

import numpy as np
import torch
from torch import Tensor
import numba
from numba import njit, prange

from decdtw import utils
from decdtw.utils import BatchedSignal
import decdtw.gdtw_solver.utils as dutils
from decdtw.gdtw_solver.nlp_solver import refine_soln_nlp
import decdtw.gdtw_solver.dp_graph as dp


def solve_gdtw(signal1: BatchedSignal, signal2: BatchedSignal, reg_wt: Union[float, Tensor],
               subseq_enabled: bool, warp_fn_times: Optional[Union[Tensor, int]] = None,
               grad_constr: Optional[Union[float, Tuple[float, float], Tensor]] = None,
               glb_constr: Optional[Union[float, Tensor]] = None, glb_lb: Optional[Tensor] = None,
               glb_ub: Optional[Tensor] = None, M: Optional[int] = None, feature_loss_p: float = 2,
               reg_loss_p: float = 2, refine_iters: int = 3, refine_factor: float = 0.125,
               refine_nlp: bool = False, mem_efficient: bool = True) -> Tuple[Tensor, BatchedSignal]:
    """
    Recovers the optimal alignment between two batched signals using GDTW with iterative refinement.

    Constructs the discretised time/warp functidutils.on value graph with node and edge costs and finds the minimum cost
    traversal using dynamic programming.

    Subsequence alignment is handled by constructing M graphs, one for each possible starting point of the
    warp function at t=0 and solves each graph traversal independently. The single best path with the lowest optimal
    cost is selected as the final alignment.

    :param signal1: Reference signal where time warp is applied
    :param signal2: Query signal where no time warp is applied. The reference signal is warped optimally to the query.
    :param reg_wt: weight applied to the warp regularisation
    :param subseq_enabled: allow subsequence alignment, i.e. no endpoint constraints
    :param warp_fn_times: Set time discretisation where the warp function is evaluated. Times are either provided directly
                          as a tensor or provide the number of uniformly sampled discrete timesteps. If None, thenTuple
                          use the same time points as signal2.
    :param glb_constr: optional. band parameters for global constraints (sakoe-chiba/R-K band parameters in [0, 1])
    :param glb_lb: optional. explicit global constraint lower bound values. Ignores glb_constr if this is provided with glb_ub
    :param glb_ub: optional. explicit global constraint upper bound values. Ignores glb_constr if this is provided with glb_lb
    :param grad_constr: optional. bounds on the time warp function derivatives (local constraints). (BxN-1) tensor input
                                  should contain UPPER bounds only with lower bounds assumed to be uniformly 0
    :param M: number of discretisations for the warp function at each time step
    :param feature_loss_p: degree of Lp loss to use (e.g. squared Euclidean p=2)
    :param reg_loss_p: degree of Lp loss on warp derivatives
    :param refine_iters: number of iterations for iterative refinement described in Deriso and Boyd
    :param refine_factor: refinement factor in [0, 1] around previous solution for each iterative refinement iteration
    :param refine_nlp: whether or not to refine DP solver solution using SLSQP
    :param mem_efficient: Generate edge weights incrementally over time, saving precious GPU memory for long time-series.
                          Applies to CUDA only, automatically handled for CPU. Memory saving is at the expense of inference time.
    :return: cost of optimal warp aligning signal1 to signal2 and corresponding warp function
    """
    assert signal1.shape[0] == signal2.shape[0], \
        f'batch size between signals not equal signal 1={signal1.shape[0]} vs signal2={signal2.shape[0]}'
    if type(warp_fn_times) != Tensor:
        assert warp_fn_times is None or warp_fn_times > 0, 'number of warp function eval times must be positive'
    else:
        assert warp_fn_times.shape[0] == signal2.shape[0], \
            f'batch size of provided warp times ({warp_fn_times.shape[0]}) differ to signals ({signal2.shape[0]})'
    assert M is None or M > 0, 'number of warp function discretisation points must be positive or None'
    device = signal1.values.device

    # if no warp function times are provided, generate automatically
    if type(warp_fn_times) == int:
        warp_fn_times = utils.batched_linspace(signal2.times[:, 0], signal2.times[:, -1], warp_fn_times)
    elif warp_fn_times is None:
        warp_fn_times = signal2.times

    # regularisation of warp derivative is away from these values
    exp_warp_grad = torch.ones((signal1.shape[0], 1), dtype=warp_fn_times.dtype, device=warp_fn_times.device)
    if not subseq_enabled:
        exp_warp_grad = signal1.times.max(dim=1, keepdim=True).values / warp_fn_times.max(dim=1, keepdim=True).values

    # setup warp fn value constraints
    if glb_lb is None or glb_ub is None:
        glb_band_lb, glb_band_ub = dutils.expand_global_constraint(glb_constr, warp_fn_times, signal1.times)
    else:
        glb_band_lb, glb_band_ub = glb_lb, glb_ub
    local_lb_vals, local_ub_vals = dutils.expand_local_constraint(grad_constr, warp_fn_times)  # gradient constraints
    local_lb_vals *= exp_warp_grad  # local constraints relative to identity slope
    local_ub_vals *= exp_warp_grad

    # automatically set solver discretisation if none provided
    if M is None:
        glb_constr = 1. if glb_constr is None else glb_constr
        max_width = glb_constr if type(glb_constr) is float else glb_constr.max()
        M = signal1.times.shape[1] if glb_constr is None else max(int(signal1.times.shape[1] * max_width) + 1, 50)

    warp_fn_lb, warp_fn_ub = dp.warp_fn_glb_bounds(warp_fn_times, signal1.times, local_lb_vals, local_ub_vals,
                                                glb_band_lb, glb_band_ub, subseq_enabled)

    reg_wt = dutils.expand_reg_wt(reg_wt, warp_fn_times)

    for i in range(refine_iters + 1):
        warp_fn_vals_dp = utils.batched_linspace(warp_fn_lb, warp_fn_ub, M)
        signal_loss_node_costs = dp.feature_node_cost(signal1, signal2, warp_fn_times, warp_fn_vals_dp, p=feature_loss_p)
        # use below instead if running into OOM issues for long time series with high dim embeddings
        # signal_loss_node_costs = dp.feature_node_cost_l2_alt(signal1, signal2, warp_fn_times, warp_fn_vals_dp)
        if not subseq_enabled:
            signal_loss_node_costs[:, 0, 1:] = np.inf
            signal_loss_node_costs[:, -1, :-1] = np.inf

        if device.type == 'cpu':
            opt_cost, opt_dp_path = solve_gdtw_with_costs_cpu(signal_loss_node_costs, warp_fn_times, warp_fn_vals_dp,
                                                              warp_fn_lb, warp_fn_ub, local_lb_vals, local_ub_vals,
                                                              exp_warp_grad, reg_wt, reg_loss_p)
        else:
            opt_cost, opt_dp_path = solve_gdtw_with_costs(signal_loss_node_costs, warp_fn_times, warp_fn_vals_dp,
                                                          local_lb_vals, local_ub_vals, exp_warp_grad, reg_wt, reg_loss_p,
                                                          mem_efficient)

        # recover actual warp fn values from optimal path indices
        opt_warp_vals = torch.gather(warp_fn_vals_dp, dim=2, index=opt_dp_path.unsqueeze(2)).squeeze(2)

        refine_warp_fn_bounds_inplace(warp_fn_lb, warp_fn_ub, opt_warp_vals, refine_factor=refine_factor)

    optimal_align_warp_fn = BatchedSignal(opt_warp_vals, times=warp_fn_times)
    if refine_nlp:
        opt_cost, opt_warp_vals_nlp = refine_soln_nlp(optimal_align_warp_fn, signal1, signal2, glb_band_lb, glb_band_ub,
                                                      local_lb_vals, local_ub_vals, reg_wt, exp_warp_grad,
                                                      feature_loss_p, reg_loss_p, subseq_enabled)
        optimal_align_warp_fn = BatchedSignal(opt_warp_vals_nlp, times=warp_fn_times)
    return opt_cost, optimal_align_warp_fn


def solve_gdtw_with_costs(node_costs: Tensor, warp_fn_times: Tensor, warp_fn_vals_dp: Tensor, grad_lb: Tensor, grad_ub: Tensor,
                          exp_warp_grad: Tensor, reg_wt: Tensor, reg_loss_p: float, mem_efficient: bool) -> Tuple[Tensor, Tensor]:
    """
    Solves the DP in the GDTW problem using precomputed node and edge costs
    """
    # compute edge costs
    reg_loss_edge_costs = dp.reg_edge_cost(warp_fn_times, warp_fn_vals_dp, grad_lb, grad_ub, exp_warp_grad, reg_wt, reg_loss_p,
                                        increm_edge=mem_efficient)

    # solve dynamic program using node and edge costs
    N = node_costs.shape[1]
    best_cost_current = node_costs[:, 0, :]
    device = best_cost_current.device
    batch_size, n_warp_discr = best_cost_current.shape
    optimal_paths = torch.zeros((batch_size, N-1, n_warp_discr), dtype=torch.long, device=device)

    for t in range(1, N):
        new_costs = best_cost_current.unsqueeze(1) + next(reg_loss_edge_costs) + node_costs[:, t, :].unsqueeze(2)
        best_cost_current, optimal_paths[:, t-1, :] = new_costs.min(dim=-1)

    optimal_path = torch.empty(size=(batch_size, N), dtype=torch.long, device=device)
    optimal_costs, optimal_path[:, -1] = best_cost_current.min(dim=1)
    if torch.any(optimal_costs > 1e20):
        raise RuntimeError("DTW solver could not find a solution, try increasing discretisation resolution or reduce constraints")
    for t in reversed(range(N-1)):
        optimal_path[:, t] = optimal_paths[torch.arange(batch_size, device=device), t, optimal_path[:, t+1]]
    return optimal_costs, optimal_path


def solve_gdtw_with_costs_cpu(node_costs: Tensor, warp_fn_times: Tensor, warp_fn_vals_dp: Tensor, warp_lb: Tensor,
                              warp_ub: Tensor, grad_lb: Tensor, grad_ub: Tensor, exp_warp_grad: Tensor,
                              reg_wt: Tensor, reg_loss_p: float) -> Tuple[Tensor, Tensor]:
    min_cost, dp_path = solve_gdtw_dp_fast_cpu(node_costs.cpu().numpy(), warp_fn_vals_dp.cpu().numpy(),
                                              torch.diff(warp_fn_times, dim=1).cpu().numpy(), warp_lb.cpu().numpy(),
                                              warp_ub.cpu().numpy(), grad_lb.cpu().numpy(), grad_ub.cpu().numpy(),
                                              exp_warp_grad.squeeze(1).cpu().numpy(), reg_wt.squeeze(1).cpu().numpy(),
                                              reg_loss_p)
    return torch.from_numpy(min_cost), torch.from_numpy(dp_path)


@njit(parallel=True, cache=True)
def solve_gdtw_dp_fast_cpu(node_costs: np.ndarray, warp_vals: np.ndarray, dts: np.ndarray, warp_lb: np.ndarray,
                          warp_ub: np.ndarray, gmin: np.ndarray, gmax: np.ndarray, exp_warp_grad: np.ndarray,
                          reg_wt: np.ndarray, p: float):
    B, N, M = node_costs.shape
    best_cost_current = node_costs[:, 0, :]
    optimal_paths = np.zeros((B, N-1, M), dtype=numba.int64)

    # run forward Bellman recursion
    for t in range(1, N):
        best_cost_temp = np.empty_like(best_cost_current)
        for b in prange(B):
            for m in range(M):
                # compute indices of connected nodes from current node
                discr_step = (warp_ub[b, t-1] - warp_lb[b, t-1]) / M
                if discr_step == 0.:
                    j_l = 0
                    j_u = 1
                else:
                    # clamp indices between 0, M-1
                    j_u = min(M-1, max((warp_vals[b, t, m] - gmin[b, t-1] * dts[b, t - 1] - warp_lb[b, t - 1]) / discr_step, 0))
                    j_l = min(M-1, max((warp_vals[b, t, m] - gmax[b, t-1] * dts[b, t - 1] - warp_lb[b, t - 1]) / discr_step, 0))
                j_l = math.floor(j_l)
                j_u = math.ceil(j_u) + 1
                # compute edge costs and cumulative costs from prev. nodes
                dwarp = (warp_vals[b, t, m] - warp_vals[b, t-1, j_l:j_u]) / dts[b, t-1]
                edge_cost = reg_wt[b] * (dwarp - exp_warp_grad[b]) ** p * dts[b, t-1]
                edge_cost[np.logical_or(dwarp < gmin[b, t-1], dwarp > gmax[b, t-1])] = np.inf
                cum_costs_to_node = best_cost_current[b, j_l:j_u] + edge_cost
                # select best cost and update paths
                idx_best = cum_costs_to_node.argmin()
                best_cost_temp[b, m] = cum_costs_to_node[idx_best] + node_costs[b, t, m]
                optimal_paths[b, t-1, m] = idx_best + j_l
        best_cost_current = best_cost_temp

    # traceback to recover path
    optimal_path = np.empty((B, N), dtype=numba.int64)
    optimal_costs = np.empty(B, dtype=node_costs.dtype)
    for b in range(B):
        idx_min = best_cost_current[b, :].argmin()
        optimal_path[b, N-1] = idx_min
        optimal_costs[b] = best_cost_current[b, idx_min]
    for t in np.flip(np.arange(N-1)):
        for b in range(B):
            optimal_path[b, t] = optimal_paths[b, t, optimal_path[b, t+1]]
    return optimal_costs, optimal_path


def refine_warp_fn_bounds_inplace(lower_bound: Tensor, upper_bound: Tensor, warp_fn_vals_centre: Tensor,
                                  refine_factor: float) -> None:
    """
    Refines upper and lower bounds for discretising the warp function at each time step around
    a given warp function. This is the core step for iterative refinement. Bounds are refined inplace.

    :param lower_bound: (BxN) initial lower bound at each time step before refinement
    :param upper_bound: (BxN) initial upper bound at each time step before refinement
    :param warp_fn_vals_centre: (BxN) position where the bound refinement will be centred
    :param refine_factor: scalar factor in (0., 1.) which determines how much bounds are refined (lower is more)
    :return: None, bounds are refined inplace
    """
    lower_proposed = warp_fn_vals_centre - 0.5 * refine_factor * (upper_bound - lower_bound)
    upper_proposed = warp_fn_vals_centre + 0.5 * refine_factor * (upper_bound - lower_bound)
    torch.maximum(lower_bound, lower_proposed, out=lower_bound)
    torch.minimum(upper_bound, upper_proposed, out=upper_bound)
    return None
    