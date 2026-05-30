"""Hybrid spin-chain solver for large entropy-vs-J2 sweeps.

The Python path is optimized for repeated exact-diagonalization scans:
- integer bitmask basis
- Numba-accelerated term assembly
- precomputed J1 and J2 operator pieces
- warm-started eigensolves across J2
- only the half-chain entropy cut is computed

The Julia path is optional and used when the chain becomes large enough that
Python exact diagonalization is no longer the best practical choice. It solves
exactly as well, but can be invoked from a single Python entry point.

For the graph in this project, the important reduction is that each curve only
needs one entropy value per J2 point: S(L/2) at half filling.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svdvals
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

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
def _cut_map_numba(states, L, ell):
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
def _term_triplets_numba(states, index_map, L, pairs, jxy, jz):
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


@njit(cache=True)
def _count_ones_u64(value):
    count = 0
    while value:
        value &= value - np.uint64(1)
        count += 1
    return count


@dataclass(frozen=True)
class SweepSpec:
    """Configuration for a single entropy-vs-J2 curve."""

    L: int
    J2_values: np.ndarray
    J1xy: float = 1.0
    J1z: float = 0.5
    J2z: float = 0.0
    periodic: bool = True


class ExactHalfFillingXXZ:
    """Python exact-diagonalization backend optimized for repeated J2 sweeps."""

    def __init__(self, L: int, Nup: int, periodic: bool = True, J1xy: float = 1.0, J1z: float = 0.5):
        self.L = int(L)
        self.Nup = int(Nup)
        self.periodic = bool(periodic)
        self.J1xy = float(J1xy)
        self.J1z = float(J1z)

        self.states = self._build_basis_states()
        self.index_map = self._build_index_map()
        self.nn_pairs = np.asarray(self._pairs(1), dtype=np.int32)
        self.nnn_pairs = np.asarray(self._pairs(2), dtype=np.int32)
        self.cut_cache: dict[int, tuple[np.ndarray, np.ndarray, int, int]] = {}

        # Precompute the operator pieces once. Sweeps only change J2.
        self.h_nn = self._term_matrix(self.nn_pairs, self.J1xy, self.J1z)
        self.h_nnn_xy = self._term_matrix(self.nnn_pairs, 1.0, 0.0)
        self.h_nnn_z = self._term_matrix(self.nnn_pairs, 0.0, 1.0)

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
        raise ValueError("order must be 1 or 2")

    def hilbert_dimension(self):
        return int(self.states.size)

    def _term_matrix(self, pairs, jxy, jz):
        rows, cols, vals = _term_triplets_numba(self.states, self.index_map, self.L, pairs, jxy, jz)
        dim = self.hilbert_dimension()
        return coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsr()

    def hamiltonian(self, j2xy: float, j2z: float = 0.0):
        return self.h_nn + float(j2xy) * self.h_nnn_xy + float(j2z) * self.h_nnn_z

    def ground_state(self, j2xy: float, j2z: float = 0.0, v0=None):
        H = self.hamiltonian(j2xy=j2xy, j2z=j2z)
        if H.shape[0] == 1:
            return float(H[0, 0]), np.array([1.0], dtype=float)

        eigvals, eigvecs = eigsh(H, k=1, which="SA", tol=1e-10, maxiter=100000, v0=v0)
        return float(eigvals[0]), eigvecs[:, 0]

    def _cut_map(self, ell: int):
        if ell in self.cut_cache:
            return self.cut_cache[ell]

        row_a, row_b = _cut_map_numba(self.states, self.L, ell)
        data = (row_a, row_b, 1 << ell, 1 << (self.L - ell))
        self.cut_cache[ell] = data
        return data

    def midchain_entropy(self, psi: np.ndarray):
        ell = self.L // 2
        row_a, row_b, dim_a, dim_b = self._cut_map(ell)
        psi_matrix = np.zeros((dim_a, dim_b), dtype=complex)
        psi_matrix[row_a, row_b] = psi
        svals = svdvals(psi_matrix)
        probs = svals**2
        probs = probs[probs > 1e-14]
        return float(-np.sum(probs * np.log(probs)))

    def sweep_entropy(self, j2_values: Sequence[float], j2z: float = 0.0):
        entropies = []
        energies = []
        psi_prev = None
        for j2 in j2_values:
            energy, psi = self.ground_state(j2xy=float(j2), j2z=float(j2z), v0=psi_prev)
            entropies.append(self.midchain_entropy(psi))
            energies.append(energy)
            psi_prev = psi
        return np.asarray(entropies, dtype=float), np.asarray(energies, dtype=float)


class JuliaHalfFillingXXZ:
    """Julia backend for the same sweep using SparseArrays + Arpack."""

    def __init__(self, julia_executable: str | None = None):
        self.julia = julia_executable or shutil.which("julia") or "julia"

    def sweep_entropy(self, spec: SweepSpec):
        j2_csv = ",".join(f"{float(v):.16g}" for v in np.asarray(spec.J2_values, dtype=float))
        script = textwrap.dedent(
            f"""
            using SparseArrays
            using LinearAlgebra
            using Arpack

            function basis_states(L::Int, Nup::Int)
                states = UInt64[]
                function rec(site::Int, chosen::Int, value::UInt64)
                    if chosen == Nup
                        push!(states, value)
                        return
                    end
                    if site > L
                        return
                    end
                    if chosen + (L - site + 1) < Nup
                        return
                    end
                    rec(site + 1, chosen + 1, value | (UInt64(1) << (L - site)))
                    rec(site + 1, chosen, value)
                end
                rec(1, 0, UInt64(0))
                return states
            end

            function bond_pairs(L::Int, order::Int, periodic::Bool)
                pairs = Vector{{Tuple{{Int,Int}}}}()
                if order == 1
                    if periodic
                        for i in 0:L-1
                            push!(pairs, (i, mod(i + 1, L)))
                        end
                    else
                        for i in 0:L-2
                            push!(pairs, (i, i + 1))
                        end
                    end
                elseif order == 2
                    if periodic
                        for i in 0:L-1
                            push!(pairs, (i, mod(i + 2, L)))
                        end
                    else
                        for i in 0:L-3
                            push!(pairs, (i, i + 2))
                        end
                    end
                else
                    error("order must be 1 or 2")
                end
                return pairs
            end

            function term_matrix(states, index, L::Int, pairs, jxy::Float64, jz::Float64)
                dim = length(states)
                rows = Int32[]
                cols = Int32[]
                vals = Float64[]
                diag = zeros(Float64, dim)
                one = UInt64(1)

                for i in eachindex(states)
                    state = states[i]
                    for (a, b) in pairs
                        pa = UInt64(L - 1 - a)
                        pb = UInt64(L - 1 - b)
                        bit_a = (state >> pa) & one
                        bit_b = (state >> pb) & one
                        if bit_a == bit_b
                            diag[i] += jz / 4.0
                        else
                            diag[i] -= jz / 4.0
                            if jxy != 0.0
                                flipped = state ⊻ ((one << pa) | (one << pb))
                                j = index[flipped]
                                push!(rows, Int32(i))
                                push!(cols, Int32(j))
                                push!(vals, jxy / 2.0)
                            end
                        end
                    end
                end

                for i in eachindex(states)
                    push!(rows, Int32(i))
                    push!(cols, Int32(i))
                    push!(vals, diag[i])
                end

                return sparse(rows, cols, vals, dim, dim)
            end

            function cut_entropy(states, psi, L::Int)
                ell = div(L, 2)
                dim_a = 1 << ell
                dim_b = 1 << (L - ell)
                row_a = Vector{{Int32}}(undef, length(states))
                row_b = Vector{{Int32}}(undef, length(states))
                left_shift = L - ell
                left_mask = UInt64(dim_a - 1)
                right_mask = UInt64(dim_b - 1)
                for i in eachindex(states)
                    state = states[i]
                    row_a[i] = Int32((state >> UInt64(left_shift)) & left_mask)
                    row_b[i] = Int32(state & right_mask)
                end
                mat = zeros(ComplexF64, dim_a, dim_b)
                @inbounds for i in eachindex(states)
                    mat[row_a[i] + 1, row_b[i] + 1] = psi[i]
                end
                svals = svdvals(mat)
                probs = svals .^ 2
                probs = probs[probs .> 1e-14]
                return -sum(probs .* log.(probs))
            end

            function ground_state(H; v0 = nothing)
                if v0 === nothing
                    vals, vecs = eigs(H; nev = 1, which = :SR)
                else
                    vals, vecs = eigs(H; nev = 1, which = :SR, v0 = v0)
                end
                return real(vals[1]), vecs[:, 1]
            end

            L = {int(spec.L)}
            Nup = L ÷ 2
            periodic = {'true' if spec.periodic else 'false'}
            J1xy = {float(spec.J1xy)}
            J1z = {float(spec.J1z)}
            J2z = {float(spec.J2z)}
            j2_values = [{j2_csv}]

            states = basis_states(L, Nup)
            index = Dict{{UInt64, Int}}()
            for (i, s) in enumerate(states)
                index[s] = i
            end
            pairs1 = bond_pairs(L, 1, periodic)
            pairs2 = bond_pairs(L, 2, periodic)

            H1 = term_matrix(states, index, L, pairs1, J1xy, J1z)
            H2xy = term_matrix(states, index, L, pairs2, 1.0, 0.0)
            H2z = term_matrix(states, index, L, pairs2, 0.0, 1.0)

            global v0 = nothing
            for j2 in j2_values
                H = H1 + j2 * H2xy + J2z * H2z
                energy, psi = ground_state(H; v0 = v0)
                entropy = cut_entropy(states, psi, L)
                println(string(j2, " ", entropy, " ", energy))
                global v0 = psi
            end
            """
        )

        try:
            proc = subprocess.run(
                [self.julia, "--startup-file=no", "-e", script],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"Julia backend failed:\n{err.stderr}") from err

        j2_out = []
        entropy_out = []
        energy_out = []
        for line in proc.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            j2_out.append(float(parts[0]))
            entropy_out.append(float(parts[1]))
            energy_out.append(float(parts[2]))
        return np.asarray(j2_out), np.asarray(entropy_out), np.asarray(energy_out)


def even_l_entropy_sweep(
    l_values: Iterable[int] = range(2, 21, 2),
    j2_min: float = 0.0,
    j2_max: float = 3.0,
    j2_step: float = 0.1,
    backend: str = "auto",
    julia_threshold: int = 20,
    periodic: bool = True,
    j1xy: float = 1.0,
    j1z: float = 0.5,
    j2z: float = 0.0,
):
    """Compute the even-L half-chain entropy sweep and return data for plotting.

    The returned dictionary maps each chain length L to a tuple:
    (J2 values, entropies, energies).
    """

    j2_values = np.round(np.arange(j2_min, j2_max + 1e-9, j2_step), 1)
    results = {}

    for L in l_values:
        if L % 2 != 0:
            continue
        Nup = L // 2

        use_julia = backend == "julia" or (backend == "auto" and L >= julia_threshold)
        if use_julia:
            solver = JuliaHalfFillingXXZ()
            spec = SweepSpec(L=L, J2_values=j2_values, J1xy=j1xy, J1z=j1z, J2z=j2z, periodic=periodic)
            j2_out, entropies, energies = solver.sweep_entropy(spec)
        else:
            chain = ExactHalfFillingXXZ(L=L, Nup=Nup, periodic=periodic, J1xy=j1xy, J1z=j1z)
            entropies, energies = [], []
            psi_prev = None
            for j2 in j2_values:
                energy, psi = chain.ground_state(j2xy=float(j2), j2z=float(j2z), v0=psi_prev)
                entropies.append(chain.midchain_entropy(psi))
                energies.append(energy)
                psi_prev = psi
            j2_out = j2_values
            entropies = np.asarray(entropies, dtype=float)
            energies = np.asarray(energies, dtype=float)

        results[L] = (np.asarray(j2_out, dtype=float), np.asarray(entropies, dtype=float), np.asarray(energies, dtype=float))

    return results


def plot_even_l_entropy_sweep(
    l_values: Iterable[int] = range(2, 21, 2),
    j2_min: float = 0.0,
    j2_max: float = 3.0,
    j2_step: float = 0.1,
    backend: str = "auto",
    julia_threshold: int = 20,
    periodic: bool = True,
    j1xy: float = 1.0,
    j1z: float = 0.5,
    j2z: float = 0.0,
    title: str | None = None,
):
    results = even_l_entropy_sweep(
        l_values=l_values,
        j2_min=j2_min,
        j2_max=j2_max,
        j2_step=j2_step,
        backend=backend,
        julia_threshold=julia_threshold,
        periodic=periodic,
        j1xy=j1xy,
        j1z=j1z,
        j2z=j2z,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for L, (j2_values, entropies, energies) in results.items():
        ax.plot(j2_values, entropies, marker="o", linewidth=2, label=f"L={L}")

    ax.set_xlabel("J2")
    ax.set_ylabel("Half-chain entanglement entropy")
    ax.set_title(title or "Half-chain entanglement entropy vs J2 for even L")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()
    return fig, ax, results


def estimated_reasonable_max_L(backend: str = "auto", julia_threshold: int = 20):
    """Rule-of-thumb limit for this exact-diagonalization workflow.

    This is intentionally conservative for the full sweep plot.
    """
    if backend == "julia":
        return 22
    if backend == "python":
        return 20
    return 22 if shutil.which("julia") else 20


if __name__ == "__main__":
    print("Recommended max L:", estimated_reasonable_max_L())
    plot_even_l_entropy_sweep(
        l_values=range(2, 21, 2),
        j2_min=0.0,
        j2_max=3.0,
        j2_step=0.1,
        backend="auto",
        julia_threshold=20,
        title="Half-chain entanglement entropy vs J2 for even L = 2..20",
    )
