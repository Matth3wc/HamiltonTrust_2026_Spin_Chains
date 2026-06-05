"""State-tracking utilities to match eigenstates across parameter steps."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_states(prev_vecs: np.ndarray, new_vecs: np.ndarray):
    """Return index mapping that matches `new_vecs` to `prev_vecs` by overlap.

    Both inputs are 2D arrays with column vectors. The function returns an
    array `perm` such that new_vecs[:, perm[i]] best matches prev_vecs[:, i].
    """
    ov = np.abs(prev_vecs.conj().T @ new_vecs)
    row_ind, col_ind = linear_sum_assignment(-ov)
    perm = np.empty(len(row_ind), dtype=int)
    perm[row_ind] = col_ind
    return perm
