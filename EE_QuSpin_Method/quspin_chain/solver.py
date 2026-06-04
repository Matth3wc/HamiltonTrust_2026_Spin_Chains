"""Diagonalization helpers."""
from typing import Tuple
import numpy as np


class Solver:
    """Provide small- and medium-scale diagonalization utilities.

    Methods:
        diagonalize(H, k=None): returns (energies, states) where `states`
        is a 2D array whose columns are eigenvectors (if dense) or a list of
        vectors for iterative solvers.
    """

    @staticmethod
    def diagonalize(H, k: int = None, which: str = "SA") -> Tuple[np.ndarray, np.ndarray]:
        try:
            # prefer sparse iterative solver for large matrices
            if k is not None:
                eigs, vecs = H.eigsh(k=k, which=which)
                return np.array(eigs), np.array(vecs)
        except Exception:
            pass

        # fallback to dense diagonalization
        mat = H.toarray()
        w, v = np.linalg.eigh(mat)
        if k is not None:
            return w[:k].copy(), v[:, :k].copy()
        return w.copy(), v.copy()
