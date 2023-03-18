import itertools
from typing import List

import numpy as np
import torch
from torch import Tensor

from scipy.linalg import solve_banded


def batched_tridiagonal_solve(A: Tensor, B: Tensor):
    """
    Solves AX=B where A has a tridiagonal structure. Faster than torch.linalg.solve for large n.

    :param A: (bxnxn) LHS tensor which is assumed to be tridiagonal for each batch element.
    :param B: (bxnxk) contains RHS (can have multiple).
    :return: (bxn) tensor containing solution X
    """
    b, n, _ = A.shape
    ab = torch.zeros((b, 3, n), dtype=A.dtype, device=A.device)
    ab[:, 0, 1:] = torch.diagonal(A, offset=1, dim1=-2, dim2=-1)
    ab[:, 1, :] = torch.diagonal(A, offset=0, dim1=-2, dim2=-1)
    ab[:, 2, :-1] = torch.diagonal(A, offset=-1, dim1=-2, dim2=-1)
    ab_np = ab.detach().cpu().numpy()
    b_np = B.detach().cpu().numpy()
    X = np.empty(shape=b_np.shape, dtype=ab_np.dtype)
    for i in range(b):
        X[i, ...] = solve_banded((1, 1), ab_np[i], b_np[i], check_finite=False, overwrite_b=True, overwrite_ab=True)
    return torch.from_numpy(X).to(A.device)


def _get_uniform_indices(constraint_deriv: List[Tensor]):
    constr_indices = sorted(range(len(constraint_deriv)), key=lambda i: constraint_deriv[i].shape[0])
    grouped_constr_ind = itertools.groupby(constr_indices, key=lambda i: constraint_deriv[i].shape[0])
    return [list(grouped[1]) for grouped in grouped_constr_ind]