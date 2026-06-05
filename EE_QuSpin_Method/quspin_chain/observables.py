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
    """Total z magnetisation in Pauli units.

    All-up state gives +L.
    All-down state gives -L.
    """
    # Compute expectation of total Pauli-Z directly from the computational
    # basis bitstrings to avoid constructing an operator repeatedly. For
    # spin-1/2 with the `pauli=True` basis convention, a basis state's bit
    # value 1 corresponds to eigenvalue +1 and 0 to -1 for sigma_z. Thus
    # the per-state magnetisation is sum_i (2*bit_i - 1).
    probs = np.abs(state) ** 2
    # build magnetisation per basis state
    L = basis.L
    # use basis.int_to_state to get bit arrays for each integer state
    ms = []
    for s in basis.states:
        # extract bits from integer representation; order matches int_to_state
        bits = ((int(s) >> np.arange(L - 1, -1, -1)) & 1)
        ms.append(float((2 * bits - 1).sum()))
    ms = np.array(ms, dtype=float)

    return float(np.dot(probs, ms))

