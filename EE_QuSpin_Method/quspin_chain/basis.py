"""Basis construction utilities using QuSpin."""
from typing import Any
from .config import ModelConfig

def build_basis(cfg: ModelConfig):
    """Build a spin-1/2 basis for the given configuration.

    This is a thin wrapper around `quspin.basis.spin_basis_1d` and forwards
    `cfg.basis_kwargs` for symmetry options (translation, parity, kblock, mblock, etc.).
    """
    try:
        from quspin.basis import spin_basis_1d
    except Exception as e:
        raise ImportError("quspin is required to build the basis") from e

    return spin_basis_1d(L=cfg.L, pauli=False, **cfg.basis_kwargs)
