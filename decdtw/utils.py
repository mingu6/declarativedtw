from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn.functional import pairwise_distance


class BatchedSignal:
    """
    Object containing a batched time series/signal. We observe the signal at a discrete set of times, and this class
    allows evaluating the signal at arbitrary continuous time points within the time domain of the signal. Currently,
    only piecewise linear interpolation has been implemented to evaluate the time series between observed times.

    This class supports 1D (useful for warp functions) and 2D (useful for time series with multiple features) signals.

    Note we assume all signals being at t=0 so observations will be standardised so that the smallest time value is 0.
    """
    def __init__(self, values: Tensor, times: Optional[Union[Tensor, float]] = None, pre_sorted: bool = True, checks: bool = True, eps: float = 1e-4):
        """
        :param times: (BxN) tensor of times where signal is observed or None (linspace (0,1), or float (max time)).
        :param values: (BxN) or (BxNxD) tensor of observed signal values. (BxN) case is a 1d signal, (BxNxD) case is 2d.
        :param pre_sorted: Whether or not the times are pre-sorted. Sorting will occur if not.
        :param checks: Performs value checking, turn off for maximum performance.
        """
        assert values.ndim in [2, 3], 'values must be BxNxD or BxN'
        Bv, Nv, *_ = values.shape
        dtype, device = values.dtype, values.device
        # todo allow for singleton dimension in time
        if type(times) == Tensor:
            assert times.ndim == 2, 'times must be BxN'
        elif type(times) == float:
            times = torch.linspace(0., times, Nv, dtype=dtype, device=device, requires_grad=False)[None, :].expand(Bv, -1)
        else:
            times = torch.linspace(0., 1., Nv, dtype=dtype, device=device, requires_grad=False)[None, :].expand(Bv, -1)
        Bt, Nt = times.shape
        assert Bt == Bv and Nt == Nv, f'times {Bt}x{Nt} do not align with values {Bv}x{Nv}'
        if not pre_sorted:
            ind_sorted = torch.argsort(times, dim=1)
            times = times[ind_sorted]
            values = values[ind_sorted]
        dt = torch.diff(times, dim=1)
        if checks and torch.count_nonzero(dt) != torch.numel(dt):
            raise ValueError("cannot allow duplicate time values")
        self.checks = checks

        self.values = values
        self.times = times - times.min(dim=1)[0][:, None]  # ensures all signals start at t=0, assumed in DTW
        self.shape = tuple(values.shape)
        self.ndim = 1 if values.ndim == 2 else 2
        self.eps = eps

    def __call__(self, times: Tensor):
        return self._interpolate(times, checks=self.checks)

    def _interpolate(self, eval_times: Tensor, checks: bool = True):
        """
        Interpolates observed signal values at provided time points. Input array can be arbitrarily sized.
        Allows for slerp interpolation but assumes (and does not check) vectors are l2-normalised. CAUTION: If
        vectors are not normalised, acos operation used in slerp may cause sneaky nans!!!

        :param eval_times: (...) Tensor containing time points where signal is to be evaluated. If (B, ...), assumes
                           eval times are provided separately per batch element.
        :param checks: Performs value checking, turn off for maximum performance.
        :return: (Bx...xD) or (Bx....) tensor of signal features evaluated at given times.
        """
        B = self.shape[0]
        eval_times1 = eval_times if eval_times.shape[0] == B else eval_times.unsqueeze(0).expand(B, *eval_times.shape)

        if checks and torch.any(eval_times1.view(B, -1).lt(self.times.min(dim=1)[0][:, None] - self.eps)):
            raise ValueError('some query points lower than interpolation range')
        if checks and torch.any(eval_times1.view(B, -1).gt(self.times.max(dim=1)[0][:, None] + self.eps)):
            raise ValueError('some query points greater than interpolation range')

        if self.ndim == 1:
            return batch_interp(eval_times1, self.times, self.values)
        else:
            return batch_interp_nd(eval_times1, self.times, self.values)

    def __repr__(self):
        repr = ' '.join([
            f'{self.__class__.__name__}:',
            f'Batch size: {self.shape[0]}',
            f'Length: {self.shape[1]}',
            f'Feature dim: {1 if self.ndim == 1 else self.shape[2]}'
        ])
        return repr

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, item):
        times = self.times[item] if type(item) != int else self.times[item].unsqueeze(0)
        values = self.values[item] if type(item) != int else self.values[item].unsqueeze(0)
        return BatchedSignal(values, times)


def dtw_objective(signal1: BatchedSignal, signal2: BatchedSignal, warp_fn: BatchedSignal, reg_wt: Union[float, Tensor],
                  exp_warp_grad: Optional[Tensor] = None, feature_loss_p: float = 2., reg_loss_p: float = 2.) -> Tensor:
    """
    Evaluate GDTW objective function given two batched signals/time series. Currently does not consider constraints.

    :param signal1: First batch of time series
    :param signal2: Second batch of time series, assumed to be the same size as the first batch in at least
                    first (batch) and last (feature dim) dimensions
    :param warp_fn: Warp functions to apply to signal1 to recover best alignment to signal2.
                    Assumed to also be of size B in the batch dimensions. One warp per element in batch,
                    corresponding to same index signals in signal1 and signal2.
    :param reg_wt: Weight applied to instantaneous regularisation loss in objective
                   (either uniform or (B,) batch element-wise)
    :param exp_warp_grad: (Bx1) Tensor or None with element-wise values to take regularisation penalty from
    :param feature_loss_p: degree of Lp loss to use (e.g. squared Euclidean p=2)
    :param reg_loss_p: Instantaneous warp regularisation function applied to time warp derivatives.
    :return: obj: (B,) tensor containing GDTW objective values for batch
    """
    B1,_,  D1 = signal1.shape
    B2, _, D2 = signal2.shape
    assert B1 == B2 and D2 == D2, f'signal1 and signal2 are incompatible, signal1 has' \
                                  f'B={B1}, D={D1} and signal2 has B={B2}, D={D2}'
    Bw = warp_fn.times.shape[0]
    assert Bw == B1, f'warp fn incompatible with signals, warp B={Bw}, signals B={B1}'
    signal_loss = signal_error(warp_fn, signal1, signal2, p=feature_loss_p)
    reg_loss = regularisation_loss(warp_fn, exp_warp_grad, reg_loss_p)
    if type(reg_wt) is float:
        reg_wt = torch.full_like(signal_loss, reg_wt)
    obj = signal_loss + reg_wt.squeeze() * reg_loss
    return obj


def regularisation_loss(warp_fn: BatchedSignal, exp_warp_deriv: Optional[Tensor], p: float) -> Tensor:
    if exp_warp_deriv is None:
        exp_warp_deriv = 1.
    dt = torch.diff(warp_fn.times, dim=1)
    dwarp = torch.diff(warp_fn.values, dim=1)
    warp_deriv = dwarp / dt
    reg_loss = ((warp_deriv - exp_warp_deriv) ** p * dt).sum(dim=1)
    return reg_loss


def signal_error(warp_fn: BatchedSignal, s1: BatchedSignal, s2: BatchedSignal, p=2.):
    losses = pairwise_distance(s1(warp_fn.values), s2(warp_fn.times), p=p) ** p
    loss = torch.trapz(losses, warp_fn.times, dim=1)
    return loss


def warp_error(warp_fn_pred: BatchedSignal, warp_fn_true: BatchedSignal, p=1.):
    warp_losses = pairwise_distance(warp_fn_pred(warp_fn_true.times).unsqueeze(2),
                                    warp_fn_true.values.unsqueeze(2), p=p) ** p
    loss = torch.trapz(warp_losses, warp_fn_true.times, dim=1)
    return loss


def batched_linspace(lower: Tensor, upper: Tensor, steps: int):
    assert lower.shape == upper.shape
    incremental = torch.linspace(0., 1., steps, device=lower.device)
    ls = lower[..., None] + (upper - lower)[..., None] * incremental[(None,) * lower.ndim + (...,)]
    return ls


def batch_interp(x: Tensor, xp: Tensor, fp: Tensor):
    """
    Simple version of np.interp with default arguments which also assumes xp is sorted. Unlike the numpy
    equivalent, this operates on batched 1d functions.

    :param x: Tensor (B, ...) containing x-coordinates at which to evaluate the interpolated values
    :param xp: Tensor (B, M) containing x-coordinates of the data points
    :param fp: Tensor (B, M) containing y-coordinates of the data points
    :return: y: Tensor the same size as x containing interpolated values
    """
    B, N = fp.shape
    i = torch.searchsorted(xp, x.view(B, -1), right=True)
    i = torch.clamp(i, min=1, max=N-1)
    im1 = i - 1
    x_im1 = torch.gather(xp, dim=1, index=im1)
    x_i = torch.gather(xp, dim=1, index=i)
    f_im1 = torch.gather(fp, dim=1, index=im1)
    f_i = torch.gather(fp, dim=1, index=i)
    y = torch.lerp(f_im1, f_i, (x.view(B, -1) - x_im1) / (x_i - x_im1))
    return y.reshape(x.shape)


def batch_interp_nd(x: Tensor, xp: Tensor, fp: Tensor):
    """
    Identical to batch_interp above, however allows teh y-coordinates to be n-dim vectors

    :param x: Tensor (B, ...) containing x-coordinates at which to evaluate the interpolated values
    :param xp: Tensor (B, M) containing x-coordinates of the data points
    :param fp: Tensor (B, M) containing y-values (vectors) of the data points
    :return: y: Tensor (B, ..., D) the same size as x except for the last dimension containing interpolated vectors
    """
    B, N, D = fp.shape
    i = torch.searchsorted(xp, x.view(B, -1), right=True)
    i = torch.clamp(i, min=1, max=N-1)
    im1 = i - 1
    x_im1 = torch.gather(xp, dim=1, index=im1).unsqueeze(2)
    x_i = torch.gather(xp, dim=1, index=i).unsqueeze(2)
    f_im1 = torch.gather(fp, dim=1, index=im1.unsqueeze(2).expand(-1, -1, D))
    f_i = torch.gather(fp, dim=1, index=i.unsqueeze(2).expand(-1, -1, D))
    f_x = torch.lerp(f_im1, f_i, (x.view(B, -1, 1) - x_im1) / (x_i - x_im1))
    return f_x.reshape(x.shape + (-1,))
