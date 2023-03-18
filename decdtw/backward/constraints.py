from typing import Tuple, List

import torch
from torch import Tensor

from decdtw.utils import BatchedSignal


def active_constraint_inds(warp_fn: BatchedSignal, signal1_times: Tensor, glb_lb: Tensor, glb_ub: Tensor,
                           lcl_grad_ub: Tensor, exp_warp_grad: Tensor, subseq: bool, eps: float) -> Tuple:
    """
    Returns indices for points in warp function where inequality constraints are active. Constraints
    come from either global constraints (e.g. RK/SC band) or when a warp gradient constraint occurs locally.
    Boundary constraints (e.g. non-negativity) may also be applicable.

    :param warp_fn: warp function to evaluate for active constraints
    :param signal1_times: (BxN) reference signal observed times to which the warp function is applied
    :param glb_lb: (BxN) time-varying global lower bound relating constraints (e.g. R-K band)
    :param glb_ub: (BxN) time-varying global upper bound relating constraints (e.g. R-K band)
    :param lcl_grad_ub: (BxN-1) time-varying warp gradient upper bounds where elem i relates to gradient from t=i to i+1
    :param exp_warp_grad: (Bx1) tensor with element-wise expected warp slope to rescale local constraints
    :param subseq: subsequence alignment enabled
    :param eps: numerical tolerance to consider constraint as active
    :return: List of indices where each element contains the indices of active inequality constraints for a single
             batch element. Lower bound constraints and upper bound constraints are returned separately. Local derivative,
             global warp function value (e.g. RK band) and boundary constraints returned separately
    """
    # global band constraints
    warp_fn_max_vals = signal1_times.max(dim=1, keepdim=True).values
    lb_glb_mask = warp_fn.values.isclose(glb_lb, rtol=0., atol=eps)
    ub_glb_mask = warp_fn.values.isclose(glb_ub, rtol=0., atol=eps)
    # do not duplicate constraints for initial/terminal points (already covered by eq. constr. in constraints_grad_hY)
    if not subseq:
        lb_glb_mask[:, 0] = False
        ub_glb_mask[:, 0] = False
        lb_glb_mask[:, -1] = False
        ub_glb_mask[:, -1] = False
    lb_glb_ind = [torch.nonzero(b).squeeze() for b in lb_glb_mask]
    ub_glb_ind = [torch.nonzero(b).squeeze() for b in ub_glb_mask]

    # local gradient constraints (assumes lower bound is uniformly 0)
    dwarp = torch.diff(warp_fn.values, dim=1)
    dt = torch.diff(warp_fn.times, dim=1)
    lb_lcl_mask = dwarp.isclose(dwarp.new_tensor([[0.]]), rtol=0., atol=eps)
    if not subseq:
        ub_lcl_mask = dwarp.isclose(lcl_grad_ub * exp_warp_grad * dt, rtol=0., atol=eps)
    else:
        ub_lcl_mask = dwarp.isclose(lcl_grad_ub * dt, rtol=0., atol=eps)
    lb_lcl_ind = [torch.nonzero(b).squeeze() for b in lb_lcl_mask]
    ub_lcl_ind = [torch.nonzero(b).squeeze() for b in ub_lcl_mask]

    # boundary constraints, i.e. [0, t_max] only if subsequence alignment enabled
    if subseq:
        lb_bnd_mask = warp_fn.values.isclose(dwarp.new_tensor([[0.]]))
        ub_bnd_mask = warp_fn.values.isclose(warp_fn_max_vals)
    else:
        lb_bnd_mask = torch.full_like(lb_glb_mask, False)  # if endpoint constraints, no boundary needed due to monotonicity
        ub_bnd_mask = torch.full_like(ub_glb_mask, False)
    lb_bnd_ind = [torch.nonzero(b).squeeze() for b in lb_bnd_mask]
    ub_bnd_ind = [torch.nonzero(b).squeeze() for b in ub_bnd_mask]
    return lb_glb_ind, ub_glb_ind, lb_lcl_ind, ub_lcl_ind, lb_bnd_ind, ub_bnd_ind


def constraints_grad_hY(grad_lb_ind: List[Tensor], grad_ub_ind: List[Tensor], glb_lb_ind: List[Tensor],
                        glb_ub_ind: List[Tensor], bnd_lb_ind: List[Tensor], bnd_ub_ind: List[Tensor],
                        N: int, subseq_enabled: bool, dtype, device) -> List[Tensor]:
    """
    Creates the A matrix for each element within a batch as specified in the DDN paper, concretely the hY term
    representing the derivative of the active constraints w.r.t. the time warp vector. Returns a list where
    each element corresponds to an A for a single batch element.

    :param grad_lb_ind: List of indices where gradient lower bound constraints are active for each batch elem.
    :param grad_ub_ind: List of indices where gradient upper bound constraints are active for each batch elem.
    :param glb_lb_ind: List of indices where global R-K lower bound constraints are active for each batch elem.
    :param glb_ub_ind: List of indices where global R-K upper bound constraints are active for each batch elem.
    :param bnd_lb_ind: List of indices where lower boundary (i.e. non-negativity) constraints are active for each batch elem.
    :param bnd_ub_ind: List of indices where boundary (i.e. max value) constraints are active for each batch elem.
    :param N: number of warp time points (decision variables)
    :param subseq_enabled: Assumes subsequence alignment, i.e. no endpoint constraints
    :param dtype: torch dtype
    :param device: torch device (cpu, cuda)
    :return: List[Tensor] where element i contains the (p_i + q_i) x M gradient matrices.
    """
    batch_hY = []
    for lcl_l, lcl_u, glb_l, glb_u, bnd_l, bnd_u in zip(grad_lb_ind, grad_ub_ind, glb_lb_ind, glb_ub_ind, bnd_lb_ind,
                                                        bnd_ub_ind):
        n_lcl_l, n_lcl_u, n_glb_l, n_glb_u, n_bnd_l, n_bnd_u = \
            torch.numel(lcl_l), torch.numel(lcl_u), torch.numel(glb_l), torch.numel(glb_u), torch.numel(
                bnd_l), torch.numel(bnd_u)
        num_active_constraints = n_lcl_l + n_lcl_u + n_glb_l + n_glb_u + n_bnd_l + n_bnd_u
        if not subseq_enabled:
            num_active_constraints += 2  # warp endpoint equality constraints
        hY = torch.zeros(size=(num_active_constraints, N), dtype=dtype, device=device)
        ind = 0
        # gradient w.r.t. warp of active local gradient constraints (lower then upper)
        lcl_l_inds = torch.arange(ind, ind + n_lcl_l, device=device)
        hY[lcl_l_inds, lcl_l] = 1.
        hY[lcl_l_inds, lcl_l + 1] = -1.
        ind += n_lcl_l
        lcl_u_inds = torch.arange(ind, ind + n_lcl_u, device=device)
        hY[lcl_u_inds, lcl_u] = -1.
        hY[lcl_u_inds, lcl_u + 1] = 1.
        ind += n_lcl_u
        # gradient w.r.t. warp of active global band constraints (lower then upper)
        glb_l_inds = torch.arange(ind, ind + n_glb_l, device=device)
        hY[glb_l_inds, glb_l] = -1.
        ind += n_glb_l
        glb_u_inds = torch.arange(ind, ind + n_glb_u, device=device)
        hY[glb_u_inds, glb_u] = 1.
        ind += n_glb_u
        # gradient w.r.t. warp of active boundary constraints (lower then upper)
        bnd_l_inds = torch.arange(ind, ind + n_bnd_l, device=device)
        hY[bnd_l_inds, bnd_l] = -1.
        ind += n_bnd_l
        bnd_u_inds = torch.arange(ind, ind + n_bnd_u, device=device)
        hY[bnd_u_inds, bnd_u] = 1.
        ind += n_bnd_u
        # encode equality constraintsdecdtw.gdtw_solver.
        if not subseq_enabled:
            hY[[-2, -1], [0, -1]] = 1.
        batch_hY.append(hY)
    return batch_hY


def constraints_grad_hGlbConstr(grad_lb_ind: List[Tensor], grad_ub_ind: List[Tensor], glb_lb_ind: List[Tensor],
                                glb_ub_ind: List[Tensor], bnd_lb_ind: List[Tensor], bnd_ub_ind: List[Tensor],
                                N: int, subseq_enabled: bool, dtype, device) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Compute gradients of active inequality constraints, e.g. phi_i \leq b_i, w.r.t global lower/upper bound values b_i.

    :param grad_lb_ind: List of indices where gradient lower bound constraints are active for each batch elem.
    :param grad_ub_ind: List of indices where gradient upper bound constraints are active for each batch elem.
    :param glb_lb_ind: List of indices where global R-K lower bound constraints are active for each batch elem.
    :param glb_ub_ind: List of indices where global R-K upper bound constraints are active for each batch elem.
    :param bnd_lb_ind: List of indices where lower boundary (i.e. non-negativity) constraints are active for each batch elem.
    :param bnd_ub_ind: List of indices where boundary (i.e. max value) constraints are active for each batch elem.
    :param N: number of warp time points (decision variables)
    :param subseq_enabled: Assumes subsequence alignment, i.e. no endpoint constraints
    :param dtype: torch dtype
    :param device: torch device (cpu, cuda)
    :return: pair of List[Tensor]s where element i contains the (p_i + q_i) x M gradient matrices for lower/upper bounds.
    """
    batch_hGlb_lo = []
    batch_hGlb_up = []
    for i, (lcl_l, lcl_u, glb_l, glb_u, bnd_l, bnd_u) in enumerate(zip(grad_lb_ind, grad_ub_ind, glb_lb_ind, glb_ub_ind,
                                                                       bnd_lb_ind, bnd_ub_ind)):
        n_lcl_l, n_lcl_u, n_glb_l, n_glb_u, n_bnd_l, n_bnd_u = \
            torch.numel(lcl_l), torch.numel(lcl_u), torch.numel(glb_l), torch.numel(glb_u), torch.numel(
                bnd_l), torch.numel(bnd_u)
        num_active_constraints = n_lcl_l + n_lcl_u + n_glb_l + n_glb_u + n_bnd_l + n_bnd_u
        if not subseq_enabled:
            num_active_constraints += 2  # warp endpoint equality constraints
        n_lcl = n_lcl_l + n_lcl_u
        hGlb_lo = torch.zeros(size=(num_active_constraints, N), dtype=dtype, device=device)
        hGlb_lo[torch.arange(n_lcl, n_lcl + n_glb_l, device=device), glb_l] = 1.
        hGlb_up = torch.zeros(size=(num_active_constraints, N), dtype=dtype, device=device)
        hGlb_up[torch.arange(n_lcl + n_glb_l, n_lcl + n_glb_l + n_glb_u, device=device), glb_u] = -1.
        batch_hGlb_lo.append(hGlb_lo), batch_hGlb_up.append(hGlb_up)
    return batch_hGlb_lo, batch_hGlb_up


def constraints_grad_hLclUbConstr(grad_lb_ind: List[Tensor], grad_ub_ind: List[Tensor], glb_lb_ind: List[Tensor],
                                  glb_ub_ind: List[Tensor], bnd_lb_ind: List[Tensor], bnd_ub_ind: List[Tensor],
                                  warp_fn_times: Tensor, exp_warp_grad: Tensor,
                                  N: int, subseq_enabled: bool, dtype, device) -> List[Tensor]:
    """
    Compute gradients of active inequality constraints w.r.t local gradient constraints (upper bound only).
    Constraints are parameterised by dwarp/dt_i <=(>=) ub_i

    :param grad_lb_ind: List of indices where gradient lower bound constraints are active for each batch elem.
    :param grad_ub_ind: List of indices where gradient upper bound constraints are active for each batch elem.
    :param glb_lb_ind: List of indices where global lower bound constraints are active for each batch elem. (e.g. RK band)
    :param glb_ub_ind: List of indices where global upper bound constraints are active for each batch elem. (e.g. RK band)
    :param bnd_lb_ind: List of indices where lower boundary (i.e. non-negativity) constraints are active for each batch elem.
    :param bnd_ub_ind: List of indices where boundary (i.e. max value) constraints are active for each batch elem.
    :param warp_fn_times: (BxN) warp function evaluation times
    :param exp_warp_grad: (Bx1) tensor with element-wise expected warp slope to rescale local constraints
    :param N: number of warp time points (decision variables)
    :param subseq_enabled: Assumes subsequence alignment, i.e. no endpoint constraints
    :param dtype: torch dtype
    :param device: torch device (cpu, cuda)
    :return: List[Tensor] where element i contains the (p_i + q_i) x M gradient matrices.
    """
    batch_hLcl = []
    dt = torch.diff(warp_fn_times, dim=1)
    for i, (lcl_l, lcl_u, glb_l, glb_u, bnd_l, bnd_u) in enumerate(zip(grad_lb_ind, grad_ub_ind, glb_lb_ind, glb_ub_ind,
                                                                       bnd_lb_ind, bnd_ub_ind)):
        n_lcl_l, n_lcl_u, n_glb_l, n_glb_u, n_bnd_l, n_bnd_u = \
            torch.numel(lcl_l), torch.numel(lcl_u), torch.numel(glb_l), torch.numel(glb_u), torch.numel(
                bnd_l), torch.numel(bnd_u)
        num_active_constraints = n_lcl_l + n_lcl_u + n_glb_l + n_glb_u + n_bnd_l + n_bnd_u
        if not subseq_enabled:
            num_active_constraints += 2  # warp endpoint equality constraints
        hLcl = torch.zeros(size=(num_active_constraints, N-1), dtype=dtype, device=device)
        hLcl[torch.arange(n_lcl_l, n_lcl_l + n_lcl_u, device=device), lcl_u] = -exp_warp_grad[i] * dt[i, lcl_u]
        batch_hLcl.append(hLcl)
    return batch_hLcl