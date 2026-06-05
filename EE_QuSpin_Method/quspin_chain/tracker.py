"""State-tracking utilities to match eigenstates across parameter steps."""
import numpy as np


def match_states(prev_vecs: np.ndarray, new_vecs: np.ndarray):
    """Return index mapping that matches `new_vecs` to `prev_vecs` by overlap.

    Both inputs are 2D arrays with column vectors. The function returns an
    array `perm` such that new_vecs[:, perm[i]] best matches prev_vecs[:, i].
    """
    # compute overlap matrix: |<prev_i | new_j>|
    ov = np.abs(prev_vecs.conj().T @ new_vecs)
    n = ov.shape[0]
    perm = -np.ones(n, dtype=int)

    # global greedy assignment: repeatedly pick the largest remaining overlap
    # and assign that (row, col) pair, then remove the row and column.
    ov_copy = ov.copy()
    for _ in range(n):
        # find index of maximum remaining overlap
        idx = ov_copy.argmax()
        i, j = divmod(int(idx), ov_copy.shape[1])
        perm[i] = j
        # invalidate selected row and column
        ov_copy[i, :] = -1
        ov_copy[:, j] = -1

    return perm
