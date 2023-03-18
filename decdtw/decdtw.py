from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

import functorch

from decdtw.gdtw_solver.dp_solver import solve_gdtw
from decdtw.utils import BatchedSignal, batched_linspace, dtw_objective
import decdtw.backward.constraints as bconstraints
import decdtw.backward.utils as butils
import decdtw.gdtw_solver.utils as dutils


class DecDTWLayer(torch.nn.Module):
    """
    DecDTW layer. Currently non-differentiable hyperparameters are stored as attributes (e.g. solver hyperparameters
    and differentiable parameters (e.g. constraints, regularisation) are provided as tensors in the forward pass.

    :param n_warp_fn_times: level of discretisation of warp function (size of warp representation)
    :param n_warp_discr: level of discretisation used in GDTW solver to estimate warp fn values
    :param subseq_enabled: do GDTW subsequence alignment instead of endpoint aligned alignment
    :param feature_loss_p: degree of norm used for computing GDTW feature loss.
    :param reg_loss_p: degree of norm used for computing GDTW warp regularisation.
    :param eps: numerical constant used to avoid div0 errors
    :param refine_iters: number of warp refinement iterations used in GDTW solver
    :param refine_factor: degree of refinement per refine iteration in GDTW solver
    :param mem_efficient: use slower but more memory efficient method for computing edge weights in the GDTW solver
    :param subseq_enabled: allow subsequence alignment, i.e. no endpoint constraints
    """
    def __init__(self, n_warp_fn_times: int = None, n_warp_discr: int = None,
                 subseq_enabled: bool = False, feature_loss_p: float = 2, reg_loss_p: float = 2,
                 eps: float = 1e-5, refine_iters: int = 3, refine_factor: float = 0.125, mem_efficient: bool = False):
        super(DecDTWLayer, self).__init__()
        self.n_warp_fn_times = n_warp_fn_times
        self.n_warp_discr = n_warp_discr
        self.subseq_enabled = subseq_enabled
        self.feature_loss_p = feature_loss_p
        self.reg_loss_p = reg_loss_p
        self.eps = eps
        self.refine_iters = refine_iters
        self.refine_factor = refine_factor
        self.mem_efficient = mem_efficient

    def dtw_objective(self, signal1: BatchedSignal, signal2: BatchedSignal, warp_fn: BatchedSignal,
                      reg_wt: Union[float, Tensor]) -> Tensor:
        return dtw_objective(signal1, signal2, warp_fn, reg_wt, self.feature_loss_p, self.reg_loss_p)

    def forward(self, signal1: BatchedSignal, signal2: BatchedSignal, reg_wt: Union[float, Tensor],
                band_constr: Optional[Union[float, Tensor]] = None, grad_constr: Optional[Union[float, Tensor]] = None,
                refine_nlp: bool = False) -> BatchedSignal:
        """
        :param signal1: Reference signal where time warp is applied
        :param signal2: Query signal where no time warp is applied. The reference signal is warped optimally to the query.
        :param reg_wt: weight applied to the warp regularisation
        :param band_constr: optional. global band parameters for global constraints (sakoe-chiba/R-K band parameters in [0, 1]).
                            Can be a single float (uniform width band) or a tensor defining band at each warp time.
        :param grad_constr: optional. upper bound for warp gradient. Can be a float (uniform bound) or tensor with time-varying bounds.
        :param refine_nlp: whether or not to use scipy's SLSQP solver to refine the DP solver estimate
        :return: opt_warp_fn: predicted warp function from GDTW as a BatchedSignal
        """
        # setup discretisation of warp function
        if self.n_warp_fn_times is None:
            warp_fn_times = signal2.times
        else:
            warp_fn_times = batched_linspace(signal2.times[:, 0], signal2.times[:, -1], self.n_warp_fn_times)
        # expand parameters for forward pass
        reg_wt = dutils.expand_reg_wt(reg_wt, warp_fn_times)
        glb_lo, glb_up = dutils.expand_global_constraint(band_constr, warp_fn_times, signal1.times)
        _,  grad_constr = dutils.expand_local_constraint(grad_constr, warp_fn_times)
        # call DTW solver with configuration
        opt_warp_fn_values = DecDTWFcn.apply(signal1.values, signal2.values, reg_wt, glb_lo, glb_up, grad_constr,
                                                     signal1.times, signal2.times, warp_fn_times, self.subseq_enabled,
                                                     self.n_warp_discr, self.feature_loss_p, self.reg_loss_p, self.eps,
                                                     self.refine_iters, self.refine_factor, refine_nlp, self.mem_efficient)
        opt_warp_fn = BatchedSignal(opt_warp_fn_values, times=warp_fn_times)
        return opt_warp_fn


class DecDTWFcn(torch.autograd.Function):
    """
    Declarative DTW forward and backward pass implemented here. Forward pass calls the DTW solver and backwards computes
    gradients w.r.t. forward pass inputs using DDN paper results. Gradients can be computed w.r.t.:

    - time series features

    - global band constraints parameterised in [0, 1]

    - local gradient constraints (upper bound only, lower bound is uniformly 0)

    - regularisation weight in the GDTW objective functions

    Gradients are computed only when necessary, i.e. requires_grad = True for input tensor.
    """
    @staticmethod
    def forward(ctx, signal1_features: Tensor, signal2_features: Tensor, reg_wt: Tensor,
                glb_lb: Tensor, glb_ub: Tensor, lcl_grad_ub: Tensor, signal1_times: Tensor, signal2_times: Tensor,
                warp_fn_times: Tensor, subseq_enabled: bool,
                n_warp_discr: int = None, feature_loss_p: float = 2., reg_loss_p: float = 2., eps=1e-5,
                refine_iters: int = 3, refine_factor: float = 0.125, refine_nlp: bool = False,
                mem_efficient: bool = False) -> Tensor:
        with torch.no_grad():
            signal1 = BatchedSignal(signal1_features, times=signal1_times)
            signal2 = BatchedSignal(signal2_features, times=signal2_times)
            _, solved_warp_fn = solve_gdtw(signal1, signal2, reg_wt, subseq_enabled,
                                           warp_fn_times=warp_fn_times, M=n_warp_discr,
                                           grad_constr=lcl_grad_ub, glb_lb=glb_lb, glb_ub=glb_ub,
                                           feature_loss_p=feature_loss_p, reg_loss_p=reg_loss_p,
                                           refine_iters=refine_iters, refine_factor=refine_factor,
                                           refine_nlp=refine_nlp, mem_efficient=mem_efficient)
        ctx.save_for_backward(signal1_features.detach(), signal2_features.detach(), solved_warp_fn.values,
                              reg_wt.detach(), glb_lb.detach(), glb_ub.detach(), lcl_grad_ub.detach(),
                              signal1_times, signal2_times, warp_fn_times)
        ctx.req_grad = [signal1_features.requires_grad, signal2_features.requires_grad, glb_lb.requires_grad,
                        glb_ub.requires_grad, lcl_grad_ub.requires_grad, reg_wt.requires_grad]
        ctx.subseq_enabled = subseq_enabled
        ctx.feature_loss_p = feature_loss_p
        ctx.reg_loss_p = reg_loss_p
        ctx.refine_iters = refine_iters
        ctx.refine_factor = refine_factor
        ctx.eps = eps
        return solved_warp_fn.values

    @staticmethod
    def backward(ctx, dJdP):
        signal1_features, signal2_features, solved_warp_fn_values, reg_wt, glb_lb, glb_ub, lcl_grad_ub, \
            signal1_times, signal2_times, solved_warp_fn_times = ctx.saved_tensors
        signal1_features.requires_grad = True
        signal2_features.requires_grad = True
        reg_wt.requires_grad = True
        solved_warp_fn = BatchedSignal(solved_warp_fn_values, times=solved_warp_fn_times)
        b, m = solved_warp_fn_values.shape
        _, n1, d = signal1_features.shape
        _, n2, _ = signal2_features.shape
        device = signal1_features.device
        dtype = signal1_features.dtype
        v = dJdP  # use vector v as alias for dJdP to simplify variable naming

        # regularisation of warp derivative is away from this value, local bounds rescaled by this value
        warp_fn_max_values = signal1_times.max(dim=1, keepdim=True).values
        exp_warp_deriv = torch.ones((b, 1), dtype=dtype, device=device)
        if not ctx.subseq_enabled:
            exp_warp_deriv = warp_fn_max_values / solved_warp_fn.times.max(dim=1, keepdim=True).values

        @torch.enable_grad()
        def dtw_objective_wrapper(signal1_times, signal2_times, exp_warp_d, warp_times, warp_fn_values,
                                  signal1_values, signal2_values, reg_wt):
            signal1_ = BatchedSignal(signal1_values.unsqueeze(0), times=signal1_times.unsqueeze(0), checks=False)
            signal2_ = BatchedSignal(signal2_values.unsqueeze(0), times=signal2_times.unsqueeze(0), checks=False)
            warp_fn_ = BatchedSignal(warp_fn_values.unsqueeze(0), times=warp_times.unsqueeze(0), checks=False)
            return dtw_objective(signal1_, signal2_, warp_fn_, reg_wt.unsqueeze(0), exp_warp_grad=exp_warp_d.unsqueeze(0),
                                 feature_loss_p=ctx.feature_loss_p, reg_loss_p=ctx.reg_loss_p).squeeze(0)

        s1, s2, lb, ub, gup, reg = ctx.req_grad
        S1JVP, S2JVP, glb_lb_JVP, glb_ub_JVP, gradJVP, regJVP = None, None, None, None, None, None

        ########### precompute matrices required for backward pass computation in prop. 4.6 in Gould et al., i.e., H, A ###########
        H = functorch.vmap(functorch.hessian(dtw_objective_wrapper, 4))(signal1_times, signal2_times, exp_warp_deriv, solved_warp_fn_times,
                                                                        solved_warp_fn_values, signal1_features, signal2_features, reg_wt)
        # Compute A from prop 4.6 in Gould et al.
        glb_lb_inds, glb_ub_inds, grad_lb_inds, grad_ub_inds, bnd_lb_inds, bnd_ub_inds = bconstraints.active_constraint_inds(
            solved_warp_fn, signal1_times, glb_lb, glb_ub, lcl_grad_ub, exp_warp_deriv, ctx.subseq_enabled, ctx.eps)
        A = bconstraints.constraints_grad_hY(grad_lb_inds, grad_ub_inds, glb_lb_inds, glb_ub_inds, bnd_lb_inds, bnd_ub_inds,
                                m, ctx.subseq_enabled, dtype, device)

        # Handle gradients w.r.t. underlying time series features and regularisation weight only if required
        if s1 or s2 or reg:
            S1JVP = torch.zeros_like(signal1_features)
            S2JVP = torch.zeros_like(signal2_features)
            vTHinv = butils.batched_tridiagonal_solve(H, v)
            # cache incremental computations of JVP from left to right for v^TDy(x) in prop. 4.6. in Gould et al.
            v_lhs = torch.zeros_like(solved_warp_fn_times)

        # Handle gradients of global constraints only if required. Relates to C matrix in prop 4.6 in Gould et al.
        if lb or ub:
            C_glb_lb, C_glb_ub = bconstraints.constraints_grad_hGlbConstr(grad_lb_inds, grad_ub_inds, glb_lb_inds, glb_ub_inds, 
                    bnd_lb_inds, bnd_ub_inds, m, ctx.subseq_enabled, dtype, device)
            glb_lb_JVP = torch.zeros_like(solved_warp_fn_times)  # b x m
            glb_ub_JVP = torch.zeros_like(solved_warp_fn_times)  # b x m
            
        # Handle gradients of local constraints only if required. Relates to C matrix in prop 4.6 in Gould et al.
        if gup:
            hGradUp = bconstraints.constrAaints_grad_hLclUbConstr(grad_lb_inds, grad_ub_inds, glb_lb_inds, glb_ub_inds, bnd_lb_inds,
                                                    bnd_ub_inds, solved_warp_fn_times, exp_warp_deriv,
                                                    m, ctx.subseq_enabled, dtype, device)
            gradJVP = torch.zeros_like(lcl_grad_ub)

        grouped_indices = butils._get_uniform_indices(A)  # pool equivalent number of active constraints together

        ########### do bulk of precomputation for JVP w.r.t. signals and JVP computation for differentiating constraints ###########
        for pooled_inds in grouped_indices:
            H_u = H[pooled_inds, ...]
            A_u = torch.stack(tuple(A[i] for i in pooled_inds), dim=0)
            b_u, p_u, _ = A_u.shape
            if p_u == 0:  # no active equality constraints, default to unconstrained gradient
                continue
            HinvAT = butils.batched_tridiagonal_solve(H_u, A_u.transpose(1, 2))  # precompute H^-1A^T block
            vTHinvAT = torch.einsum('bm,bmn->bn', dJdP[pooled_inds], HinvAT)  # compute v^TH^-1A^T
            AHinvAT = torch.bmm(A_u, HinvAT)  # precompute AH^-1A^T

            ########### compute v^TH^-1A^T(AH^-1A^t)^-1 ###########

            # try:  # cholesky solve first... numerically unstable sometimes and may yield nans, thar be dragons here!
            #     AHinvAT_decomp = torch.linalg.cholesky(AHinvAT, upper=False)
            #     v_lhs[pooled_inds, :] = torch.cholesky_solve(v2.unsqueeze(2), AHinvAT_decomp, upper=False).squeeze(2)  # b_u x p_u
            # except:
            try:  # do full LU for solve
                gnarly_thing = torch.linalg.solve(AHinvAT, vTHinvAT)  # v^TH^-1A^T(AH^-1A^t)^-1
            except:  # screw it, least squares???? Maybe not regular
                gnarly_thing = torch.linalg.lstsq(AHinvAT, vTHinvAT).solution

            ############ update JVP for constraint parameters, i.e., Dy(x) = -v^TH^-1A^T(AH^-1A^T)^-1C for the subset of x which relates to constraints      ############
            ############ note, other terms for Dy(x) in prop. 4.6 in Gould et al. are zero, since B = 0 for constraint params b/c they are not in obj. fn. f ############

            if lb:
                glb_lb_JVP[pooled_inds] = -torch.einsum('bp,bpn->bn', gnarly_thing, torch.stack(tuple(hGlbl[i] for i in pooled_inds), dim=0))
            if ub:
                glb_ub_JVP[pooled_inds] = -torch.einsum('bp,bpn->bn', gnarly_thing, torch.stack(tuple(hGlbu[i] for i in pooled_inds), dim=0))
            if gup:
                gradJVP[pooled_inds] = -torch.einsum('bp,bpn->bn', gnarly_thing, torch.stack(tuple(hGradUp[i] for i in pooled_inds), dim=0))
            
            ############ compute v^TH^-1A^T(AH^-1A^t)^-1AH^-1 ############

            if s1 or s2 or reg:
                v_lhs[pooled_inds, :] = torch.einsum('bp,bpm->bm', gnarly_thing, HinvAT.transpose(1, 2))

        ############ compute JVP = ( v^TH^-1A^T(AH^-1A^t)^-1AH^-1 - v^TH^-1 )B. note, terms in brackets are precomputed in above code ############
        ############ note, this only applies to S1, S2, lambda since C = 0 for these parameters (do not appear in constraint fns)     ############

        def vhp_s1(s1t, s2t, exp_d, warp_t, warp, s1f,  s2f, rg, tangents):
            fY = functorch.grad(dtw_objective_wrapper, 4)
            return functorch.vjp(lambda x: fY(s1t, s2t, exp_d, warp_t, warp, x, s2f, rg), s1f)[1](tangents)

        def vhp_s2(s1t, s2t, exp_d, warp_t, warp, s1f,  s2f, rg, tangents):
            fY = functorch.grad(dtw_objective_wrapper, 4)
            return functorch.vjp(lambda x: fY(s1t, s2t, exp_d, warp_t, warp, s1f, x, rg), s2f)[1](tangents)

        def vhp_reg(s1t, s2t, exp_d, warp_t, warp, s1f,  s2f, rg, tangents):
            fY = functorch.grad(dtw_objective_wrapper, 4)
            return functorch.vjp(lambda x: fY(s1t, s2t, exp_d, warp_t, warp, s1f, s2f, x), rg)[1](tangents)

        # only compute JVP if requires_grad is True for variable
        if s1:
            S1JVP = functorch.vmap(vhp_s1)(signal1_times, signal2_times, exp_warp_deriv, solved_warp_fn_times,
                                           solved_warp_fn_values, signal1_features, signal2_features, reg_wt, v_lhs - vTHinv)[0]
        if s2:
            S2JVP = functorch.vmap(vhp_s2)(signal1_times, signal2_times, exp_warp_deriv, solved_warp_fn_times,
                                           solved_warp_fn_values, signal1_features, signal2_features, reg_wt, v_lhs - vTHinv)[0]
        if reg:
            regJVP = functorch.vmap(vhp_reg)(signal1_times, signal2_times, exp_warp_deriv, solved_warp_fn_times,
                                             solved_warp_fn_values, signal1_features, signal2_features, reg_wt, v_lhs - vTHinv)[0]
        return S1JVP, S2JVP, regJVP, glb_lb_JVP, glb_ub_JVP, gradJVP, None, None, None, None, None, None, None, None, None, None, None, None
