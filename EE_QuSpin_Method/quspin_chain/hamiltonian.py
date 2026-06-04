"""Hamiltonian builder for NN and NNN XXZ models using QuSpin."""
from typing import Optional
from .config import ModelConfig

def _bond_list(L, offset, pbc):
    if pbc:
        return [[i, (i + offset) % L] for i in range(L)]
    else:
        return [[i, i + offset] for i in range(L - offset)]


def build_hamiltonian(cfg: ModelConfig, basis, dtype=None):
    """Construct an XXZ Hamiltonian with optional NNN terms.

    Returns a QuSpin `hamiltonian` object.
    """
    try:
        from quspin.operators import hamiltonian
    except Exception as e:
        raise ImportError("quspin is required to build Hamiltonians") from e

    # NN terms
    nn_xy = [[cfg.Jxy, i, j] for i, j in _bond_list(cfg.L, 1, cfg.pbc)]
    nn_z = [[cfg.Jz, i, j] for i, j in _bond_list(cfg.L, 1, cfg.pbc)]

    static = [
        ["xx", nn_xy],
        ["yy", nn_xy],
        ["zz", nn_z],
    ]

    # NNN terms (offset 2)
    if cfg.Jxy2 != 0.0 or cfg.Jz2 != 0.0:
        nnn_xy = [[cfg.Jxy2, i, j] for i, j in _bond_list(cfg.L, 2, cfg.pbc)]
        nnn_z = [[cfg.Jz2, i, j] for i, j in _bond_list(cfg.L, 2, cfg.pbc)]
        static += [["xx", nnn_xy], ["yy", nnn_xy], ["zz", nnn_z]]

    import numpy as _np
    if dtype is None:
        dtype = _np.float64
    H = hamiltonian(static, [], basis=basis, dtype=dtype)
    return H
