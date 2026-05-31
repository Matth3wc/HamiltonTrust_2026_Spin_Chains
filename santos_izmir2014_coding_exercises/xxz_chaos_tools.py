from __future__ import annotations

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


@njit

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


@njit

def _build_hamiltonian_triplets(states, index_map, L, pairs1, pairs2, J1xy, J1z, J2xy, J2z, kind_flag):
    dim = states.shape[0]
    n1 = pairs1.shape[0]
    n2 = pairs2.shape[0]
    if kind_flag == 0:
        max_nnz = dim + dim * n1
    else:
        max_nnz = dim + dim * (n1 + n2)

    rows = np.empty(max_nnz, dtype=np.int32)
    cols = np.empty(max_nnz, dtype=np.int32)
    vals = np.empty(max_nnz, dtype=np.float64)
    diag = np.zeros(dim, dtype=np.float64)
    one = np.uint64(1)
    cursor = 0

    for i in range(dim):
        state = states[i]

        for p in range(n1):
            a = pairs1[p, 0]
            b = pairs1[p, 1]
            pa = np.uint64(L - 1 - a)
            pb = np.uint64(L - 1 - b)
            bit_a = (state >> pa) & one
            bit_b = (state >> pb) & one
            if bit_a == bit_b:
                diag[i] += J1z / 4.0
            else:
                diag[i] -= J1z / 4.0
                if J1xy != 0.0:
                    flipped = state ^ ((one << pa) | (one << pb))
                    j = index_map[np.int64(flipped)]
                    rows[cursor] = i
                    cols[cursor] = j
                    vals[cursor] = J1xy / 2.0
                    cursor += 1

        if kind_flag == 1:
            for p in range(n2):
                a = pairs2[p, 0]
                b = pairs2[p, 1]
                pa = np.uint64(L - 1 - a)
                pb = np.uint64(L - 1 - b)
                bit_a = (state >> pa) & one
                bit_b = (state >> pb) & one
                if bit_a == bit_b:
                    diag[i] += J2z / 4.0
                else:
                    diag[i] -= J2z / 4.0
                    if J2xy != 0.0:
                        flipped = state ^ ((one << pa) | (one << pb))
                        j = index_map[np.int64(flipped)]
                        rows[cursor] = i
                        cols[cursor] = j
                        vals[cursor] = J2xy / 2.0
                        cursor += 1

    for i in range(dim):
        rows[cursor] = i
        cols[cursor] = i
        vals[cursor] = diag[i]
        cursor += 1

    return rows[:cursor], cols[:cursor], vals[:cursor]


class FastXXZChain:
    """Exact fixed-Nup XXZ chain with entanglement, ETH, and chaos diagnostics."""

    def __init__(self, L, Nup, periodic=True, J1xy=1.0, J1z=0.5, J2xy=0.0, J2z=0.0):
        self.L = int(L)
        self.Nup = int(Nup)
        self.periodic = bool(periodic)
        self.J1xy = float(J1xy)
        self.J1z = float(J1z)
        self.J2xy = float(J2xy)
        self.J2z = float(J2z)

        self.states = self._build_basis_states()
        self.index_map = self._build_index_map()
        self.nn_pairs = np.asarray(self._pairs(1), dtype=np.int32)
        self.nnn_pairs = np.asarray(self._pairs(2), dtype=np.int32)
        self.cut_cache = {}
        self.orbit_cache = None
        self._build_static_terms()

    def _build_static_terms(self):
        self.h_nn = self._term_matrix(self.nn_pairs, self.J1xy, self.J1z)
        self.h_nnn_xy = self._term_matrix(self.nnn_pairs, 1.0, 0.0)
        self.h_nnn_z = self._term_matrix(self.nnn_pairs, 0.0, 1.0)

    def set_couplings(self, J1xy=None, J1z=None, J2xy=None, J2z=None):
        if J1xy is not None:
            self.J1xy = float(J1xy)
        if J1z is not None:
            self.J1z = float(J1z)
        if J2xy is not None:
            self.J2xy = float(J2xy)
        if J2z is not None:
            self.J2z = float(J2z)
        self._build_static_terms()
        return self

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

    def _pairs(self, order):
        if order == 1:
            if self.periodic:
                return [(i, (i + 1) % self.L) for i in range(self.L)]
            return [(i, i + 1) for i in range(self.L - 1)]
        if order == 2:
            if self.periodic:
                return [(i, (i + 2) % self.L) for i in range(self.L)]
            return [(i, i + 2) for i in range(self.L - 2)]
        raise ValueError('order must be 1 or 2')

    def hilbert_dimension(self):
        return int(self.states.size)

    def _term_matrix(self, pairs, jxy, jz):
        rows, cols, vals = _build_hamiltonian_triplets(self.states, self.index_map, self.L, pairs, pairs, jxy, jz, 0.0, 0.0, 0)
        dim = self.hilbert_dimension()
        return coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsr()

    def build_hamiltonian(self, kind='nnn'):
        if kind == 'nn':
            return self.h_nn
        if kind != 'nnn':
            raise ValueError("kind must be 'nn' or 'nnn'.")
        return self.h_nn + self.J2xy * self.h_nnn_xy + self.J2z * self.h_nnn_z

    def hamiltonian(self, kind='nnn'):
        return self.build_hamiltonian(kind=kind)

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

    def _momentum_basis(self, k):
        if not self.periodic:
            raise ValueError('Momentum-sector diagonalisation requires periodic boundary conditions.')

        k = int(k) % self.L
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
            raise ValueError(f'No momentum basis states exist for k={k}.')

        return np.column_stack(columns)

    def _momentum_sector_hamiltonian(self, kind='nnn', momentum_sector=0):
        U = self._momentum_basis(momentum_sector)
        H = self.build_hamiltonian(kind=kind)
        Hk = U.conj().T @ (H @ U)
        Hk = 0.5 * (Hk + Hk.conj().T)
        return Hk, U

    def eigensystem(self, kind='nnn', momentum_sector=None, dense_limit=4096):
        if momentum_sector is None:
            H = self.build_hamiltonian(kind=kind)
            if H.shape[0] > dense_limit:
                raise ValueError(
                    f'Full eigensystem is too large for dense diagonalisation (dim={H.shape[0]}). '
                    'Use a smaller chain or a momentum sector.'
                )
            evals, evecs = eigh(H.toarray())
            return evals, evecs

        Hk, U = self._momentum_sector_hamiltonian(kind=kind, momentum_sector=momentum_sector)
        evals, vecs_k = eigh(Hk)
        vecs = U @ vecs_k
        return evals, vecs

    def ground_state(self, kind='nnn', momentum_sector=None, dense_limit=4096):
        if momentum_sector is None:
            H = self.build_hamiltonian(kind=kind)
            if H.shape[0] == 1:
                return float(H[0, 0]), np.array([1.0], dtype=complex)
            if H.shape[0] <= dense_limit:
                evals, evecs = eigh(H.toarray())
                return float(evals[0]), evecs[:, 0]
            evals, evecs = eigsh(H, k=1, which='SA', tol=1e-12, maxiter=100000)
            return float(evals[0]), evecs[:, 0]

        evals, evecs = self.eigensystem(kind=kind, momentum_sector=momentum_sector, dense_limit=dense_limit)
        return float(evals[0]), evecs[:, 0]

    def _cut_map(self, block_size):
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

    def _state_matrix(self, state, block_size):
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

    def entanglement_entropy_from_state(self, state, block_size):
        psi_matrix = self._state_matrix(state, block_size)
        rho_a = psi_matrix @ psi_matrix.conj().T
        evals = eigvalsh(rho_a)
        evals = np.clip(np.real(evals), 0.0, None)
        evals = evals[evals > 1e-14]
        return float(-np.sum(evals * np.log(evals)))

    def entanglement_spectrum(self, state, block_size):
        psi_matrix = self._state_matrix(state, block_size)
        rho_a = psi_matrix @ psi_matrix.conj().T
        evals = eigvalsh(rho_a)
        evals = np.clip(np.real(evals), 0.0, None)
        evals = evals[evals > 1e-14]
        return evals[::-1]

    def entanglement_energies(self, state, block_size):
        lambdas = self.entanglement_spectrum(state, block_size)
        return -np.log(lambdas)

    def eigenstate_entropies(self, block_size, kind='nnn', momentum_sector=None):
        evals, evecs = self.eigensystem(kind=kind, momentum_sector=momentum_sector)
        entropies = np.array(
            [self.entanglement_entropy_from_state(evecs[:, i], block_size) for i in range(evecs.shape[1])],
            dtype=float,
        )
        return evals, entropies

    def eth_fluctuations(self, block_size, kind='nnn', momentum_sector=None):
        evals, entropies = self.eigenstate_entropies(block_size, kind=kind, momentum_sector=momentum_sector)
        if entropies.size < 2:
            return evals, np.array([], dtype=float)
        return evals[:-1], np.diff(entropies)

    def level_spacing_ratios(self, kind='nnn', momentum_sector=None):
        evals, _ = self.eigensystem(kind=kind, momentum_sector=momentum_sector)
        spacings = np.diff(evals)
        if spacings.size < 2:
            return np.array([], dtype=float)
        return np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])

    def mean_r_statistic(self, kind='nnn', momentum_sector=None):
        r_values = self.level_spacing_ratios(kind=kind, momentum_sector=momentum_sector)
        return float(np.mean(r_values)) if r_values.size else np.nan

    def inverse_participation_ratio(self, state):
        state = np.asarray(state, dtype=complex).reshape(-1)
        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError('state must be non-zero.')
        state = state / norm
        return float(np.sum(np.abs(state) ** 4))

    def time_evolve(self, psi0, t, kind='nnn'):
        H = self.build_hamiltonian(kind=kind)
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
        return np.array([self.time_evolve(psi0, tt, kind=kind) for tt in times])

    def quench_entanglement_growth(self, psi0, times, block_size, kind='nnn'):
        times = np.asarray(times, dtype=float).reshape(-1)
        entropies = np.array(
            [self.entanglement_entropy_from_state(self.time_evolve(psi0, tt, kind=kind), block_size) for tt in times],
            dtype=float,
        )
        return times, entropies

    def chaos_diagnostics(self, J2_values, block_size, kind='nnn', momentum_sector=None):
        J2_values = np.asarray(J2_values, dtype=float).reshape(-1)
        original_J2xy = self.J2xy

        results = {
            'J2_values': J2_values,
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

        try:
            for J2 in J2_values:
                self.J2xy = float(J2)
                evals, evecs = self.eigensystem(kind=kind, momentum_sector=momentum_sector)
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
        finally:
            self.J2xy = original_J2xy

        return {key: np.asarray(value, dtype=float) if key != 'J2_values' else value for key, value in results.items()}
