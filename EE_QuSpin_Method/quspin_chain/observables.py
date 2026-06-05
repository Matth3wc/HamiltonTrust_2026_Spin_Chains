"""Observable calculators: magnetisation, fidelity, shannon, IPR, r-statistics."""
import numpy as np
from typing import Sequence


def shannon_entropy(state: np.ndarray, base: float = np.e) -> float:
    probs = np.abs(state) ** 2
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs)) / (np.log(base))


def inverse_participation_ratio(state: np.ndarray) -> float:
    probs = np.abs(state) ** 2
    return float(np.sum(probs ** 2))


def fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    return float(np.abs(np.vdot(state_a, state_b)))


def level_spacings(energies: Sequence[float]) -> np.ndarray:
    e = np.sort(np.array(energies))
    return np.diff(e)


def r_statistic(energies: Sequence[float]) -> float:
    s = level_spacings(energies)
    if len(s) < 2:
        return float('nan')
    r = np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])
    return float(np.mean(r))


def entanglement_entropy(basis, state: np.ndarray, sub_sys_A=None, density: bool = False, alpha: float = 1.0) -> float:
    """Compute von Neumann (alpha=1) or Renyi entanglement entropy for subsystem A.

    Uses `basis.ent_entropy` and returns the Sent_A value (natural log base).
    """
    res = basis.ent_entropy(state, sub_sys_A=sub_sys_A, density=density, alpha=alpha)
    # ent_entropy returns dict with key 'Sent_A'
    return float(res.get("Sent_A", res.get("Sent", float('nan'))))


def magnetisation(basis, state: np.ndarray) -> float:
    """Total S^z magnetisation (sum over sites) for a given state vector.

    The operator used is the single-site 'z' operator available via `quspin.operators.hamiltonian`.
    """
    try:
        from quspin.operators import hamiltonian
    except Exception as e:
        raise ImportError("quspin is required to build magnetisation operator") from e

    static = [["z", [[1.0, i] for i in range(basis.L)]]]
    Sz = hamiltonian(static, [], basis=basis)
    # compute expectation and use real part to avoid ComplexWarning
    val = np.vdot(state, Sz.dot(state))

    # Determine scaling so that an all-up product basis vector maps to
    # magnetisation == L. This handles QuSpin conventions where per-site
    # S^z is either +/-1 (Pauli) or +/-1/2 (spin-1/2 operator).
    try:
        Ns = int(basis.Ns)
    except Exception:
        Ns = None

    scale = 1.0
    if Ns is not None and Ns > 0:
        e0 = np.zeros(Ns, dtype=complex)
        e0[0] = 1.0
        val0 = np.vdot(e0, Sz.dot(e0)).real
        if abs(val0) > 1e-12:
            scale = float(basis.L) / float(val0)

    return float(np.real(val) * scale)

