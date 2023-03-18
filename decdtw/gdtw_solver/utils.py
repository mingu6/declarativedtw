from typing import Tuple, Optional, Union

import numpy as np
import torch
from torch import Tensor

from decdtw import utils


def expand_global_constraint(params: Optional[Union[float, Tensor]], warp_fn_times: Tensor, signal1_times: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Expands global constraints on warp function into Tensor form for solver. This interface allows ease of use
    when constraints are both learnable/fixed and flexible/uniform.

    :param params: None for no constraint, float in [0, 1] for uniform Sakoe-Chiba band constraint, (B,) or (B, 1) Tensor for
                   element-wise global constraints or (Q,) or (1, Q) or (B, Q) tensor of time-varying constraints, where the warp
                   function bounds are defined at Q uniform points and interpolation occurs in between
    :param warp_fn_times: Warp function evaluation times
    :param signal1_times: Reference (non-warped) signal times
    :return: glb_lo, glb_up are tensors shaped (B, N) containing lower/upper bounds for warp values at evaluation times
    """
    # expand param first to at least Bx2 tensor of warp knot points
    B, N = warp_fn_times.shape
    dtype, device = warp_fn_times.dtype, warp_fn_times.device
    exp_warp_grad = (signal1_times[:, -1] / warp_fn_times[:, -1]).unsqueeze(1)
    if params is None:
        params = 1.  # no constraint
    if type(params) is float:
        params = torch.full((B, 2), params, dtype=dtype, device=device)
    elif type(params) is Tensor and params.shape[0] == B:  # element-wise global band (B,) or (B, 1)
        params = params.expand_as(exp_warp_grad).expand(-1, 2)
    else:  # time-dependent R-K bands (B, n_bands) or (1, n_bands) or (n_bands)
        params = torch.atleast_2d(params).expand(B, -1)
    # define global constraint at desired knot points and interpolate warp_fn_times to them
    t_glb_up = utils.batched_linspace(warp_fn_times[:, 0], warp_fn_times[:, -1], params.shape[-1])
    glb_lo = utils.batch_interp(warp_fn_times, t_glb_up + params * signal1_times[:, -1, None], exp_warp_grad * t_glb_up)
    glb_up = utils.batch_interp(warp_fn_times, t_glb_up, exp_warp_grad * t_glb_up + params * signal1_times[:, -1, None])
    return glb_lo, glb_up


def expand_local_constraint(params: Optional[Union[float, Tuple[float, float], Tensor]],
                            warp_fn_times: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    Expands local gradient constraints on warp function into Tensor form for solver. This interface allows ease of use
    when constraints are both learnable/fixed and flexible/uniform.

    :param params: None for (0, inf) monotonicity constraint, float for monotonicity w/uniform upper bound constraint (0, upper),
                   Tuple for uniform constraints in form (lower, upper), Tensor (B,) for element-wise constraints,
                   Tensor 1xN-1 or BxN-1 for time dependent constraints (upper bound only)
    :param warp_fn_times: Optional. Warp function eval times BxN if uniform constraints used
    :return: (lower, upper) where each is a Tensor BxN-1 containing local gradient constraints for all durations
    """
    dt = torch.diff(warp_fn_times, dim=1)
    if type(params) is Tensor or type(params) is torch.nn.Parameter:
        return torch.zeros_like(dt), params.clone().expand_as(dt)
    if params is None:
        params = (0., np.inf)  # no constraint
    if type(params) is float:
        return torch.zeros_like(dt), torch.full_like(dt, params)
    elif type(params) is tuple:
        return torch.full_like(dt, params[0]), torch.full_like(dt, params[1])
    raise ValueError(f'band constraint is not a valid type: {type(params)}')


def expand_reg_wt(params: Union[float, Tensor], warp_fn_times: Tensor) -> Tensor:
    """
    Expands regularisation term in objective function to Tensor for solver. This interface allows ease of use when
    regularisation weight is a learnable parameter

    :param params: float for fixed regularisation weight or Tensor of regularisation weights
    :param warp_fn_times: (BxN) tensor containing warp function times
    :return: (B,) tensor containing per-element regularisation weights
    """
    warp_max = warp_fn_times.max(dim=1, keepdim=True).values
    if type(params) is float:
        return torch.full_like(warp_max, params)
    else:
        return params.expand_as(warp_max)


def index_and_set_inf(x: Tensor, mask: torch.BoolTensor) -> None:
    """
    Equivalent to indexing tensor with a boolean mask and setting true elements in the mask to
    inf. For large tensors this method is both faster and much more memory intensive than
    directly indexing with the boolean mask. Modifies the tensor in-place.
    """
    assert x.shape == mask.shape
    mask = mask.to(x.dtype)
    mask *= float('inf')
    torch.nan_to_num(mask, nan=0., out=mask)  # mask will have zeros, 0. * inf = nan, causes problems
    x += mask


def adjust_node_costs_trapezoidal_inplace(node_costs: Tensor, warp_fn_times: Tensor) -> None:
    """
    Applies trapezoidal adjustment to node costs. Similar to torch.trapz but performs the adjustment
    on a per element basis as opposed to summing over the time dimension. Adjustment is done inplace.

    :param node_costs: (BxNx...) node costs to which trapezoidal adjustment is applied elementwise
    :param warp_fn_times: (BxN) warp function times associated with node costs used to determine adjustment.
    :return: None, adjustment to node_costs is performed inplace
    """
    dtime = torch.diff(warp_fn_times, dim=1)
    per_node_trapz_time_adj = torch.zeros_like(warp_fn_times)
    per_node_trapz_time_adj[:, :-1] += dtime
    per_node_trapz_time_adj[:, 1:] += dtime
    node_costs *= per_node_trapz_time_adj.unsqueeze(-1)
    node_costs *= 0.5
    return None
