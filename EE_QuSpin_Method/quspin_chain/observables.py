"""Observable calculators: magnetisation, fidelity, shannon, IPR, r-statistics."""
from typing import Sequence

import numpy as np


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


def _state_norm(state: np.ndarray) -> float:
    norm = float(np.real_if_close(np.vdot(state, state)))
    return norm if norm > 0 else 1.0


def _spin_operator(basis):
    from quspin.operators import hamiltonian

    static = [["z", [[1.0, i] for i in range(basis.L)]]]
    return hamiltonian(
        static,
        [],
        basis=basis,
        dtype=np.float64,
        check_herm=False,
        check_symm=False,
        check_pcon=False,
    )


def _basis_total_magnetisation_values(basis, pauli_units: bool = True) -> np.ndarray:
    if basis.Ns != 2 ** basis.L:
        raise ValueError(
            "absolute magnetisation requires the full computational basis; "
            "use magnetisation_z_squared for symmetry-reduced bases"
        )

    states = np.asarray(basis.states, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(basis.L - 1, -1, -1)) & 1)
    values = np.sum(1 - 2 * bits, axis=1).astype(float)
    if not pauli_units:
        values *= 0.5
    return values


def magnetisation_z(basis, state: np.ndarray, per_site: bool = False, pauli_units: bool = True) -> float:
    """Return <sum_i sigma_i^z> (total) or the per-site average when ``per_site=True``.

    The basis is constructed with ``pauli=False``, so QuSpin's ``"z"`` operator
    measures spin-1/2 ``S^z``. When ``pauli_units=True`` the result is converted
    to Pauli units by multiplying by 2.
    """
    op = _spin_operator(basis)
    norm = _state_norm(state)
    value = np.vdot(state, op.dot(state)) / norm
    value = float(np.real_if_close(value))
    if pauli_units:
        value *= 2.0
    if per_site:
        value /= basis.L
    return float(value)


def magnetisation_z_abs(basis, state: np.ndarray, per_site: bool = False, pauli_units: bool = True) -> float:
    """Return <|M_z|> in computational basis when that interpretation is safe.

    For symmetry-reduced bases, the computational-basis probabilities are not
    available directly and a clear error is raised instead of returning a
    misleading value.
    """
    probs = np.abs(np.asarray(state)) ** 2
    probs = probs / probs.sum()
    values = np.abs(_basis_total_magnetisation_values(basis, pauli_units=pauli_units))
    value = float(np.dot(probs, values))
    if per_site:
        value /= basis.L
    return float(value)


def magnetisation_z_squared(basis, state: np.ndarray, per_site: bool = False, pauli_units: bool = True) -> float:
    """Return <M_z^2> (total), or per-site value when ``per_site=True``.

    The returned value is in Pauli units when ``pauli_units=True``.
    """
    op = _spin_operator(basis)
    norm = _state_norm(state)
    mz_state = op.dot(state)
    value = np.vdot(mz_state, mz_state) / norm
    value = float(np.real_if_close(value))
    if pauli_units:
        value *= 4.0
    if per_site:
        value /= basis.L ** 2
    return float(value)


def magnetisation(basis, state: np.ndarray) -> float:
    """Total magnetisation in spin units."""
    return magnetisation_z(basis, state, per_site=False, pauli_units=False)
