"""State-tracking utilities to match eigenstates across parameter steps."""
import numpy as np


def match_states(prev_vecs: np.ndarray, new_vecs: np.ndarray):
    """Return index mapping that matches `new_vecs` to `prev_vecs` by overlap.

    Both inputs are 2D arrays with column vectors. The function returns an
    array `perm` such that new_vecs[:, perm[i]] best matches prev_vecs[:, i].
    """
    # compute overlap matrix: |<prev_i | new_j>|
    ov = np.abs(prev_vecs.conj().T @ new_vecs)
    # greedy matching by maximum overlap
    n = ov.shape[0]
    perm = -np.ones(n, dtype=int)
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        j = ov[i].argmax()
        # if already used, pick next best
        k = 0
        while used[j]:
            ov[i, j] = -1
            j = ov[i].argmax()
            k += 1
            if k > n:
                break
        perm[i] = j
        used[j] = True
    return perm
