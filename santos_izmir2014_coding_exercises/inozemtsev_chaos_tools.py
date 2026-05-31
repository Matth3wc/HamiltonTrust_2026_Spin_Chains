from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import eigh, eigvalsh, svdvals
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, expm_multiply

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


@njit(cache=True)
def _build_cut_map_numba(states, L, ell):
    dim = states.shape[0]
    row_a = np.empty(dim, dtype=np.int32)
    row_b = np.empty(dim, dtype=np.int32)
    left_shift = L - ell
    left_mask = (np.uint64(1) << np.uint64(left_shift)) - np.uint64(1)
    right_mask = (np.uint64(1) << np.uint64(ell)) - np.uint64(1)

    for i in range(dim):
        state = states[i]
        row_a[i] = int((state >> np.uint64(left_shift)) & left_mask)
        row_b[i] = int(state & right_mask)

    return row_a, row_b


@njit(cache=True)
def _build_pair_triplets_numba(states, index_map, L, pairs, jxy, jz):
    dim = states.shape[0]
    nbonds = pairs.shape[0]
    max_nnz = dim + dim * nbonds

    rows = np.empty(max_nnz, dtype=np.int32)
    cols = np.empty(max_nnz, dtype=np.int32)
    vals = np.empty(max_nnz, dtype=np.float64)
    diag = np.zeros(dim, dtype=np.float64)
    one = np.uint64(1)
    cursor = 0

    for i in range(dim):
        state = states[i]
        for p in range(nbonds):
            a = pairs[p, 0]
            b = pairs[p, 1]
            pa = np.uint64(L - 1 - a)
            pb = np.uint64(L - 1 - b)
            bit_a = (state >> pa) & one
            bit_b = (state >> pb) & one
            if bit_a == bit_b:
                diag[i] += jz / 4.0
            else:
                diag[i] -= jz / 4.0
                if jxy != 0.0:
                    flipped = state ^ ((one << pa) | (one << pb))
                    j = index_map[np.int64(flipped)]
                    rows[cursor] = i
                    cols[cursor] = j
                    vals[cursor] = jxy / 2.0
                    cursor += 1

    for i in range(dim):
        rows[cursor] = i
        cols[cursor] = i
        vals[cursor] = diag[i]
        cursor += 1

    return rows[:cursor], cols[:cursor], vals[:cursor]


@dataclass(frozen=True)
class InozemtsevSweepSpec:
    """Configuration for shell-truncation sweeps."""

    L: int
    max_shell_values: np.ndarray
    alpha: float = 0.65
    normalize_shells: bool = True
    periodic: bool = True


class InozemtsevChain:
    """Finite Inozemtsev-style long-range Heisenberg chain with shell truncation.

    The model used here is a practical exact-diagonalization version of the periodic
    Inozemtsev chain:

        H = sum_{r=1}^{R} J_r sum_i S_i · S_{i+r}

    with hyperbolic shell weights

        J_r = 1 / sinh(alpha * r)^2

    optionally normalized so J_1 = 1.

    Truncating the shell range R provides a controlled way to remove terms and
    observe how spectral and entanglement diagnostics evolve away from the full
    long-range model.
    """

    def __init__(
        self,
        L: int,
        Nup: int,
        periodic: bool = True,
        alpha: float = 0.65,
        normalize_shells: bool = True,
    ):
        self.L = int(L)
        self.Nup = int(Nup)
        self.periodic = bool(periodic)
        self.alpha = float(alpha)
        self.normalize_shells = bool(normalize_shells)

        self.states = self._build_basis_states()
        self.index_map = self._build_index_map()
        self.max_unique_shell = self.L // 2 if self.periodic else self.L - 1
        self.cut_cache: dict[int, dict[str, np.ndarray | int]] = {}
        self.orbit_cache = None
        self.shell_pairs = [self._pairs(shell) for shell in range(1, self.max_unique_shell + 1)]
        self.shell_matrices = [self._shell_matrix(shell) for shell in range(1, self.max_unique_shell + 1)]
        self.shell_weights = np.array([self.shell_strength(shell) for shell in range(1, self.max_unique_shell + 1)], dtype=float)

    def _build_basis_states(self):
        states = []
        for positions in combinations(range(self.L), self.Nup):
            value = 0
            for site in positions:
                value |= 1 << (self.L - 1 - site)
            states.append(value)
        return np.asarray(states, dtype=np.uint64)

    def _build_index_map(self):
        size = 1 << self.L
        index_map = -np.ones(size, dtype=np.int32)
        for i, state in enumerate(self.states):
            index_map[int(state)] = i
        return index_map

    def hilbert_dimension(self):
        return int(self.states.size)

    def shell_strength(self, shell: int):
        shell = int(shell)
        if shell < 1:
            raise ValueError('shell must be at least 1')
        weight = 1.0 / np.sinh(self.alpha * shell) ** 2
        if self.normalize_shells:
            return float(weight / (1.0 / np.sinh(self.alpha) ** 2))
        return float(weight)

    def _pairs(self, shell: int):
        shell = int(shell)
        if shell < 1:
            raise ValueError('shell must be at least 1')

        if self.periodic:
            if shell > self.max_unique_shell:
                return []
            if self.L % 2 == 0 and shell == self.L // 2:
                return [(i, i + shell) for i in range(self.L // 2)]
            return [(i, (i + shell) % self.L) for i in range(self.L)]

        if shell >= self.L:
            return []
        return [(i, i + shell) for i in range(self.L - shell)]

    def _shell_matrix(self, shell: int):
        pairs = np.asarray(self._pairs(shell), dtype=np.int32)
        if pairs.size == 0:
            dim = self.hilbert_dimension()
            return coo_matrix((dim, dim), dtype=np.float64).tocsr()
        rows, cols, vals = _build_pair_triplets_numba(self.states, self.index_map, self.L, pairs, 1.0, 1.0)
        dim = self.hilbert_dimension()
        return coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsr()

    def build_hamiltonian(self, max_shell: int | None = None):
        if max_shell is None:
            max_shell = self.max_unique_shell
        max_shell = max(1, min(int(max_shell), self.max_unique_shell))

        H = None
        for shell in range(1, max_shell + 1):
            contribution = self.shell_weights[shell - 1] * self.shell_matrices[shell - 1]
            H = contribution if H is None else H + contribution
        return H

    def eigensystem(self, max_shell: int | None = None, momentum_sector: int | None = None, dense_limit: int = 4096):
        H = self.build_hamiltonian(max_shell=max_shell)
        if momentum_sector is None:
            if H.shape[0] > dense_limit:
                raise ValueError(
                    f'Full eigensystem is too large for dense diagonalisation (dim={H.shape[0]}). '
                    'Use a smaller chain or a momentum sector.'
                )
            evals, evecs = eigh(H.toarray())
            return evals, evecs

        Hk, U = self._momentum_sector_hamiltonian(H, momentum_sector)
        evals, vecs_k = eigh(Hk)
        vecs = U @ vecs_k
        return evals, vecs

    def ground_state(self, max_shell: int | None = None, momentum_sector: int | None = None, dense_limit: int = 4096):
        H = self.build_hamiltonian(max_shell=max_shell)
        if momentum_sector is None:
            if H.shape[0] == 1:
                return float(H[0, 0]), np.array([1.0], dtype=complex)
            if H.shape[0] <= dense_limit:
                evals, evecs = eigh(H.toarray())
                return float(evals[0]), evecs[:, 0]
            evals, evecs = eigsh(H, k=1, which='SA', tol=1e-12, maxiter=100000)
            return float(evals[0]), evecs[:, 0]

        evals, evecs = self.eigensystem(max_shell=max_shell, momentum_sector=momentum_sector, dense_limit=dense_limit)
        return float(evals[0]), evecs[:, 0]

    def _translate_state(self, state, shift):
        shift = int(shift) % self.L
        state = int(state)
        if shift == 0:
            return state
        translated = 0
        for site in range(self.L):
            if (state >> (self.L - 1 - site)) & 1:
                new_site = (site + shift) % self.L
                translated |= 1 << (self.L - 1 - new_site)
        return translated

    def _orbit_representation(self, state):
        images = [self._translate_state(state, shift) for shift in range(self.L)]
        rep = min(images)
        period = 1
        while period < self.L and self._translate_state(rep, period) != rep:
            period += 1
        return rep, period

    def _orbit_data(self):
        if self.orbit_cache is not None:
            return self.orbit_cache

        orbit_data = {}
        seen = set()
        for state in self.states:
            rep, period = self._orbit_representation(int(state))
            if rep in seen:
                continue
            seen.add(rep)
            members = [self._translate_state(rep, shift) for shift in range(period)]
            orbit_data[rep] = {'period': period, 'members': members}

        self.orbit_cache = orbit_data
        return orbit_data

    def _momentum_basis(self, momentum_sector: int):
        if not self.periodic:
            raise ValueError('Momentum-sector diagonalisation requires periodic boundary conditions.')

        k = int(momentum_sector) % self.L
        orbit_data = self._orbit_data()
        columns = []

        for rep in sorted(orbit_data):
            info = orbit_data[rep]
            period = info['period']
            if (k * period) % self.L != 0:
                continue

            vec = np.zeros(self.hilbert_dimension(), dtype=complex)
            norm = np.sqrt(period)
            for shift, member in enumerate(info['members']):
                idx = self.index_map[int(member)]
                vec[idx] = np.exp(-2j * np.pi * k * shift / self.L) / norm
            columns.append(vec)

        if not columns:
            raise ValueError(f'No momentum basis states exist for k={momentum_sector}.')

        return np.column_stack(columns)

    def _momentum_sector_hamiltonian(self, H, momentum_sector: int):
        U = self._momentum_basis(momentum_sector)
        Hk = U.conj().T @ (H @ U)
        Hk = 0.5 * (Hk + Hk.conj().T)
        return Hk, U

    def _cut_map(self, block_size: int):
        block_size = int(block_size)
        if block_size in self.cut_cache:
            return self.cut_cache[block_size]
        if not (0 < block_size < self.L):
            raise ValueError('block_size must satisfy 0 < block_size < L.')

        row_a, row_b = _build_cut_map_numba(self.states, self.L, block_size)
        mapping = {
            'rowA': row_a,
            'rowB': row_b,
            'dimA': 1 << block_size,
            'dimB': 1 << (self.L - block_size),
        }
        self.cut_cache[block_size] = mapping
        return mapping

    def _state_matrix(self, state, block_size: int):
        state = np.asarray(state, dtype=complex).reshape(-1)
        dim = self.hilbert_dimension()
        if state.size != dim:
            raise ValueError(f'state has length {state.size}, expected {dim}.')

        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError('state must be non-zero.')
        state = state / norm

        data = self._cut_map(block_size)
        psi_matrix = np.zeros((data['dimA'], data['dimB']), dtype=complex)
        psi_matrix[data['rowA'], data['rowB']] = state
        return psi_matrix

    def entanglement_entropy_from_state(self, state, block_size: int):
        psi_matrix = self._state_matrix(state, block_size)
        rho_a = psi_matrix @ psi_matrix.conj().T
        evals = eigvalsh(rho_a)
        evals = np.clip(np.real(evals), 0.0, None)
        evals = evals[evals > 1e-14]
        return float(-np.sum(evals * np.log(evals)))

    def entanglement_spectrum(self, state, block_size: int):
        psi_matrix = self._state_matrix(state, block_size)
        rho_a = psi_matrix @ psi_matrix.conj().T
        evals = eigvalsh(rho_a)
        evals = np.clip(np.real(evals), 0.0, None)
        evals = evals[evals > 1e-14]
        return evals[::-1]

    def entanglement_energies(self, state, block_size: int):
        lambdas = self.entanglement_spectrum(state, block_size)
        return -np.log(lambdas)

    def eigenstate_entropies(self, block_size: int, max_shell: int | None = None, momentum_sector: int | None = None):
        evals, evecs = self.eigensystem(max_shell=max_shell, momentum_sector=momentum_sector)
        entropies = np.array(
            [self.entanglement_entropy_from_state(evecs[:, i], block_size) for i in range(evecs.shape[1])],
            dtype=float,
        )
        return evals, entropies

    def eth_fluctuations(self, block_size: int, max_shell: int | None = None, momentum_sector: int | None = None):
        evals, entropies = self.eigenstate_entropies(block_size, max_shell=max_shell, momentum_sector=momentum_sector)
        if entropies.size < 2:
            return evals, np.array([], dtype=float)
        return evals[:-1], np.diff(entropies)

    def level_spacing_ratios(self, max_shell: int | None = None, momentum_sector: int | None = None):
        evals, _ = self.eigensystem(max_shell=max_shell, momentum_sector=momentum_sector)
        spacings = np.diff(evals)
        if spacings.size < 2:
            return np.array([], dtype=float)
        return np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])

    def mean_r_statistic(self, max_shell: int | None = None, momentum_sector: int | None = None):
        r_values = self.level_spacing_ratios(max_shell=max_shell, momentum_sector=momentum_sector)
        return float(np.mean(r_values)) if r_values.size else np.nan

    def inverse_participation_ratio(self, state):
        state = np.asarray(state, dtype=complex).reshape(-1)
        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError('state must be non-zero.')
        state = state / norm
        return float(np.sum(np.abs(state) ** 4))

    def time_evolve(self, psi0, t, max_shell: int | None = None):
        H = self.build_hamiltonian(max_shell=max_shell)
        psi0 = np.asarray(psi0, dtype=complex).reshape(-1)
        if psi0.size != self.hilbert_dimension():
            raise ValueError(f'psi0 has length {psi0.size}, expected {self.hilbert_dimension()}.')

        norm = np.linalg.norm(psi0)
        if norm == 0:
            raise ValueError('psi0 must be non-zero.')
        psi0 = psi0 / norm

        if np.isscalar(t):
            return expm_multiply((-1j * float(t)) * H, psi0)

        times = np.asarray(t, dtype=float).reshape(-1)
        return np.array([self.time_evolve(psi0, tt, max_shell=max_shell) for tt in times])

    def quench_entanglement_growth(self, psi0, times, block_size: int, max_shell: int | None = None):
        times = np.asarray(times, dtype=float).reshape(-1)
        entropies = np.array(
            [self.entanglement_entropy_from_state(self.time_evolve(psi0, tt, max_shell=max_shell), block_size) for tt in times],
            dtype=float,
        )
        return times, entropies

    def chaos_diagnostics(self, max_shell_values, block_size: int, momentum_sector: int | None = None):
        max_shell_values = np.asarray(max_shell_values, dtype=int).reshape(-1)
        results = {
            'max_shell_values': max_shell_values,
            'ground_energy': [],
            'mid_energy': [],
            'gap': [],
            'ground_entropy': [],
            'mid_entropy': [],
            'entropy_variance': [],
            'r_statistic': [],
            'ipr': [],
            'mean_spacing': [],
        }

        for max_shell in max_shell_values:
            evals, evecs = self.eigensystem(max_shell=int(max_shell), momentum_sector=momentum_sector)
            entropies = np.array(
                [self.entanglement_entropy_from_state(evecs[:, i], block_size) for i in range(evecs.shape[1])],
                dtype=float,
            )
            spacings = np.diff(evals)
            if spacings.size >= 2:
                r_values = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
                r_mean = float(np.mean(r_values))
            else:
                r_mean = np.nan

            mid_idx = len(evals) // 2
            results['ground_energy'].append(float(evals[0]))
            results['mid_energy'].append(float(evals[mid_idx]))
            results['gap'].append(float(evals[1] - evals[0]) if len(evals) > 1 else np.nan)
            results['ground_entropy'].append(float(entropies[0]))
            results['mid_entropy'].append(float(entropies[mid_idx]))
            results['entropy_variance'].append(float(np.var(entropies)))
            results['r_statistic'].append(r_mean)
            results['ipr'].append(self.inverse_participation_ratio(evecs[:, mid_idx]))
            results['mean_spacing'].append(float(np.mean(spacings)) if spacings.size else np.nan)

        return {key: np.asarray(value, dtype=float) if key != 'max_shell_values' else value for key, value in results.items()}
