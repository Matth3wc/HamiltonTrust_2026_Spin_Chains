"""
Sparse XXZ / NNN-XXZ Santos comparison.

This is a rewritten version of the original notebook-style script.

Main upgrades:
- Sparse Hamiltonians using scipy.sparse.
- Sparse diagonalisation using scipy.sparse.linalg.eigsh.
- Optional Numba acceleration for Hamiltonian construction.
- tqdm progress bars.
- Optional joblib parallelism.
- Optional QuSpin Hamiltonian builder.
- Optional quimb entropy helper.

The model is

    H =
        J1xy sum_NN  (Sx_i Sx_j + Sy_i Sy_j)
      + J1z  sum_NN  Sz_i Sz_j
      + J2xy sum_NNN (Sx_i Sx_j + Sy_i Sy_j)
      + J2z  sum_NNN Sz_i Sz_j

Your original Delta parameter corresponds to J1z.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from IPython.display import HTML, display

    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

try:
    import quimb as qu

    QUIMB_AVAILABLE = True
except Exception:
    QUIMB_AVAILABLE = False

try:
    from quspin.operators import hamiltonian as quspin_hamiltonian
    from quspin.basis import spin_basis_1d

    QUSPIN_AVAILABLE = True
except Exception:
    QUSPIN_AVAILABLE = False


try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass


# ============================================================
# Utility functions
# ============================================================

def make_pairs(
    L: int,
    distance: int,
    periodic: bool = True,
    unique_pairs: bool = True,
) -> List[Tuple[int, int]]:
    """Construct interaction pairs for a one-dimensional spin chain."""
    L = int(L)
    distance = int(distance)

    pairs: List[Tuple[int, int]] = []
    seen = set()

    if periodic:
        site_range = range(L)
    else:
        site_range = range(L - distance)

    for i in site_range:
        j = (i + distance) % L if periodic else i + distance

        if i == j:
            continue

        if unique_pairs:
            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            seen.add(key)

        pairs.append((i, j))

    return pairs


def normalise_state(state: np.ndarray) -> np.ndarray:
    """Return a normalised copy of a state vector."""
    state = np.asarray(state, dtype=complex)
    norm = np.linalg.norm(state)

    if norm == 0:
        raise ValueError("Cannot normalise the zero vector.")

    return state / norm


def fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Pure-state fidelity |<a|b>|."""
    state_a = normalise_state(state_a)
    state_b = normalise_state(state_b)
    return float(np.abs(np.vdot(state_a, state_b)))


# ============================================================
# Sparse pair-matrix construction
# ============================================================

if NUMBA_AVAILABLE:

    @njit
    def _pair_terms_coo_numba(L: int, pairs: np.ndarray):
        """Numba-accelerated COO construction for Hxy and Hzz."""
        dim = 1 << L
        n_pairs = pairs.shape[0]

        max_xy_nnz = dim * n_pairs

        rows_xy = np.empty(max_xy_nnz, dtype=np.int64)
        cols_xy = np.empty(max_xy_nnz, dtype=np.int64)
        data_xy = np.empty(max_xy_nnz, dtype=np.float64)
        diag_zz = np.zeros(dim, dtype=np.float64)

        count = 0

        for state in range(dim):
            diag_value = 0.0

            for p in range(n_pairs):
                left = pairs[p, 0]
                right = pairs[p, 1]

                bit_left = L - 1 - left
                bit_right = L - 1 - right

                occ_left = (state >> bit_left) & 1
                occ_right = (state >> bit_right) & 1

                if occ_left == occ_right:
                    diag_value += 0.25
                else:
                    diag_value -= 0.25

                    flipped = state ^ ((1 << bit_left) | (1 << bit_right))

                    rows_xy[count] = state
                    cols_xy[count] = flipped
                    data_xy[count] = 0.5
                    count += 1

            diag_zz[state] = diag_value

        return rows_xy[:count], cols_xy[:count], data_xy[:count], diag_zz

else:
    _pair_terms_coo_numba = None


def _pair_terms_coo_python(
    L: int,
    pairs: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure-Python fallback COO construction for Hxy and Hzz."""
    dim = 1 << L

    rows_xy = []
    cols_xy = []
    data_xy = []
    diag_zz = np.zeros(dim, dtype=float)

    for state in range(dim):
        diag_value = 0.0

        for left, right in pairs:
            bit_left = L - 1 - left
            bit_right = L - 1 - right

            occ_left = (state >> bit_left) & 1
            occ_right = (state >> bit_right) & 1

            if occ_left == occ_right:
                diag_value += 0.25
            else:
                diag_value -= 0.25

                flipped = state ^ ((1 << bit_left) | (1 << bit_right))

                rows_xy.append(state)
                cols_xy.append(flipped)
                data_xy.append(0.5)

        diag_zz[state] = diag_value

    return (
        np.asarray(rows_xy, dtype=np.int64),
        np.asarray(cols_xy, dtype=np.int64),
        np.asarray(data_xy, dtype=float),
        diag_zz,
    )


def build_pair_matrices(
    L: int,
    pairs: Sequence[Tuple[int, int]],
    use_numba: bool = True,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Build sparse Hxy and Hzz matrices for a given pair list.

    Hxy represents S_i^+ S_j^- + S_i^- S_j^+.
    Hzz represents S_i^z S_j^z.
    """
    dim = 1 << L

    if len(pairs) == 0:
        zero = sp.csr_matrix((dim, dim), dtype=float)
        return zero, zero

    if use_numba and NUMBA_AVAILABLE:
        pair_array = np.asarray(pairs, dtype=np.int64)
        rows_xy, cols_xy, data_xy, diag_zz = _pair_terms_coo_numba(L, pair_array)
    else:
        rows_xy, cols_xy, data_xy, diag_zz = _pair_terms_coo_python(L, pairs)

    Hxy = sp.coo_matrix(
        (data_xy, (rows_xy, cols_xy)),
        shape=(dim, dim),
        dtype=float,
    ).tocsr()

    Hzz = sp.diags(
        diag_zz,
        offsets=0,
        shape=(dim, dim),
        format="csr",
    )

    return Hxy, Hzz


# ============================================================
# Sparse XXZ / NNN-XXZ chain class
# ============================================================

@dataclass
class SparseXXZChain:
    """
    Sparse exact-diagonalisation class for the XXZ / NNN-XXZ chain.

    Original XXZ model:
        SparseXXZChain(L=6, J1xy=1.0, J1z=Delta, J2xy=0.0, J2z=0.0)

    NNN-XXZ model:
        SparseXXZChain(L=6, J1xy=1.0, J1z=Delta, J2xy=..., J2z=...)
    """

    L: int
    J1xy: float = 1.0
    J1z: float = 0.0
    J2xy: float = 0.0
    J2z: float = 0.0
    periodic: bool = True
    unique_pairs: bool = True
    use_numba: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        self.L = int(self.L)
        self.dim = 1 << self.L

        self.nn_pairs = make_pairs(
            self.L,
            distance=1,
            periodic=self.periodic,
            unique_pairs=self.unique_pairs,
        )

        self.nnn_pairs = make_pairs(
            self.L,
            distance=2,
            periodic=self.periodic,
            unique_pairs=self.unique_pairs,
        )

        if self.verbose:
            print(f"Building sparse operators for L={self.L}, dim={self.dim}")
            print(f"NN pairs:  {self.nn_pairs}")
            print(f"NNN pairs: {self.nnn_pairs}")

        self.H1xy, self.H1zz = build_pair_matrices(
            self.L,
            self.nn_pairs,
            use_numba=self.use_numba,
        )

        self.H2xy, self.H2zz = build_pair_matrices(
            self.L,
            self.nnn_pairs,
            use_numba=self.use_numba,
        )

        self.mz_values_spin_half = np.array(
            [state.bit_count() - self.L / 2 for state in range(self.dim)],
            dtype=float,
        )

    @property
    def Delta(self) -> float:
        """Compatibility with the old notation Delta = J1z."""
        return self.J1z

    @Delta.setter
    def Delta(self, value: float) -> None:
        self.J1z = float(value)

    def hamiltonian(
        self,
        J1xy: Optional[float] = None,
        J1z: Optional[float] = None,
        J2xy: Optional[float] = None,
        J2z: Optional[float] = None,
    ) -> sp.csr_matrix:
        """Construct the sparse Hamiltonian for the current couplings."""
        if J1xy is None:
            J1xy = self.J1xy
        if J1z is None:
            J1z = self.J1z
        if J2xy is None:
            J2xy = self.J2xy
        if J2z is None:
            J2z = self.J2z

        H = sp.csr_matrix((self.dim, self.dim), dtype=float)

        if J1xy != 0.0:
            H = H + float(J1xy) * self.H1xy
        if J1z != 0.0:
            H = H + float(J1z) * self.H1zz
        if J2xy != 0.0:
            H = H + float(J2xy) * self.H2xy
        if J2z != 0.0:
            H = H + float(J2z) * self.H2zz

        return H.tocsr()

    def diagonalize(
        self,
        k: Optional[int] = 5,
        which: str = "SA",
        dense: bool = False,
        return_eigenvectors: bool = True,
        tol: float = 1e-10,
        maxiter: Optional[int] = None,
    ):
        """
        Diagonalise the Hamiltonian.

        Uses scipy.sparse.linalg.eigsh by default.
        Falls back to dense diagonalisation if eigsh fails.
        """
        H = self.hamiltonian()

        use_dense = dense or (k is None) or (k >= self.dim - 1)

        if use_dense:
            H_dense = H.toarray()

            if return_eigenvectors:
                eigvals, eigvecs = np.linalg.eigh(H_dense)

                if k is not None:
                    eigvals = eigvals[:k]
                    eigvecs = eigvecs[:, :k]

                return eigvals, eigvecs

            eigvals = np.linalg.eigvalsh(H_dense)

            if k is not None:
                eigvals = eigvals[:k]

            return eigvals

        try:
            eigvals, eigvecs = spla.eigsh(
                H,
                k=k,
                which=which,
                return_eigenvectors=True,
                tol=tol,
                maxiter=maxiter,
            )

            order = np.argsort(eigvals)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            if return_eigenvectors:
                return eigvals, eigvecs

            return eigvals

        except Exception as err:
            warnings.warn(
                f"eigsh failed with error: {err}. Falling back to dense diagonalisation."
            )

            return self.diagonalize(
                k=k,
                dense=True,
                return_eigenvectors=return_eigenvectors,
            )

    def ground_state(self) -> Tuple[float, np.ndarray]:
        """Return ground-state energy and ground-state vector."""
        eigvals, eigvecs = self.diagonalize(k=1, which="SA")
        return float(eigvals[0]), eigvecs[:, 0]

    def magnetization(
        self,
        state: np.ndarray,
        per_site: bool = False,
        pauli_units: bool = False,
    ) -> float:
        """
        Compute <M_z>.

        Default units are spin-half units, so each site contributes +/- 1/2.

        If pauli_units=True, each site contributes +/- 1.
        If per_site=True, divide by L.
        """
        state = normalise_state(state)
        probabilities = np.abs(state) ** 2

        value = float(np.dot(probabilities, self.mz_values_spin_half))

        if pauli_units:
            value *= 2.0

        if per_site:
            value /= self.L

        return value

    def half_chain_entropy(
        self,
        state: np.ndarray,
        use_quimb: bool = False,
    ) -> float:
        """Half-chain von Neumann entanglement entropy."""
        state = normalise_state(state)

        if use_quimb and QUIMB_AVAILABLE:
            try:
                return float(
                    qu.entropy_subsys(
                        state,
                        dims=[2] * self.L,
                        sysa=tuple(range(self.L // 2)),
                        base=np.e,
                    )
                )
            except Exception:
                pass

        ell = self.L // 2

        psi_matrix = state.reshape(
            (1 << ell, 1 << (self.L - ell))
        )

        singular_values = np.linalg.svd(
            psi_matrix,
            compute_uv=False,
        )

        probabilities = singular_values**2
        probabilities = probabilities[probabilities > 1e-14]

        return float(-np.sum(probabilities * np.log(probabilities)))


# ============================================================
# Delta sweep functions
# ============================================================

def sweep_delta(
    chain: SparseXXZChain,
    delta_values: np.ndarray,
    k_energy: int = 5,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Sweep over Delta and compute:
    - first k_energy eigenvalues
    - ground-state magnetisation
    - ground-state half-chain entropy
    - ground-state vectors
    - fidelity between neighbouring ground states on the negative-Delta branch
    """
    energy_levels = []
    mz_values = []
    entropy_values = []
    ground_states = []

    iterator = tqdm(delta_values, desc="Delta sweep") if show_progress else delta_values

    for delta in iterator:
        chain.Delta = float(delta)

        eigvals, eigvecs = chain.diagonalize(k=k_energy, which="SA")
        ground_state = eigvecs[:, 0]

        energy_levels.append(eigvals[:k_energy])
        ground_states.append(ground_state)
        mz_values.append(chain.magnetization(ground_state))
        entropy_values.append(chain.half_chain_entropy(ground_state))

    energy_levels = np.asarray(energy_levels, dtype=float)
    mz_values = np.asarray(mz_values, dtype=float)
    entropy_values = np.asarray(entropy_values, dtype=float)
    ground_states = np.asarray(ground_states, dtype=complex)

    zero_matches = np.where(np.isclose(delta_values, 0.0))[0]

    if len(zero_matches) > 0:
        zero_idx = int(zero_matches[0])
    else:
        zero_idx = int(np.argmin(np.abs(delta_values)))

    delta_fidelity = delta_values[1:zero_idx]

    fidelity_values = np.array(
        [
            fidelity(ground_states[i - 1], ground_states[i])
            for i in range(1, zero_idx)
        ],
        dtype=float,
    )

    return {
        "delta_values": delta_values,
        "energy_levels": energy_levels,
        "mz_values": mz_values,
        "entropy_values": entropy_values,
        "ground_states": ground_states,
        "delta_fidelity": delta_fidelity,
        "fidelity_values": fidelity_values,
    }


def plot_santos_2d_results(results: Dict[str, np.ndarray], L: int) -> None:
    """Four-panel Santos-style comparison plot."""
    delta_values = results["delta_values"]
    energy_levels = results["energy_levels"]
    mz_values = results["mz_values"]
    entropy_values = results["entropy_values"]
    delta_fidelity = results["delta_fidelity"]
    fidelity_values = results["fidelity_values"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex="col")
    fig.patch.set_facecolor("white")

    panel_color = "#1f2937"
    accent_mz = "#d97706"
    accent_fidelity = "#059669"
    accent_entropy = "#dc2626"
    level_colors = ["#2563eb", "#f97316", "#16a34a", "#ef4444", "#8b5cf6"]

    n_levels = energy_levels.shape[1]

    for level in range(n_levels):
        axes[0, 0].plot(
            delta_values,
            energy_levels[:, level],
            label=f"E{level}",
            linewidth=2,
            color=level_colors[level % len(level_colors)],
        )

    axes[0, 0].set_title("First eigenvalues vs Delta", color=panel_color)
    axes[0, 0].set_ylabel("Energy")
    axes[0, 0].legend(ncol=3, frameon=False)
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].set_facecolor("#f8fafc")

    axes[0, 1].plot(
        delta_values,
        mz_values,
        color=accent_mz,
        linewidth=2,
    )
    axes[0, 1].set_title("Ground-state total $M_z$ vs Delta", color=panel_color)
    axes[0, 1].set_ylabel("$M_z$")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].set_facecolor("#f8fafc")

    axes[1, 0].plot(
        delta_fidelity,
        fidelity_values,
        color=accent_fidelity,
        linewidth=2,
    )
    axes[1, 0].set_title("Ground-state fidelity vs Delta", color=panel_color)
    axes[1, 0].set_xlabel("Delta")
    axes[1, 0].set_ylabel("Fidelity")

    if len(delta_fidelity) > 0:
        axes[1, 0].set_xlim(float(np.min(delta_fidelity)), 0.0)

    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].set_facecolor("#f8fafc")

    axes[1, 1].plot(
        delta_values,
        entropy_values,
        color=accent_entropy,
        linewidth=2,
    )
    axes[1, 1].set_title(
        "Ground-state half-chain entanglement entropy vs Delta",
        color=panel_color,
    )
    axes[1, 1].set_xlabel("Delta")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].set_facecolor("#f8fafc")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(
        f"Sparse periodic XXZ model at L={L}",
        fontsize=15,
        fontweight="semibold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================================================
# Delta-J2z sweep functions
# ============================================================

def compute_j2z_row(
    L: int,
    j2z: float,
    delta_values: np.ndarray,
    k_energy: int = 5,
    J1xy: float = 1.0,
    J2xy: float = 0.0,
    periodic: bool = True,
    unique_pairs: bool = True,
    use_numba: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute one fixed-J2z row over all Delta values."""
    chain = SparseXXZChain(
        L=L,
        J1xy=J1xy,
        J1z=float(delta_values[0]),
        J2xy=J2xy,
        J2z=float(j2z),
        periodic=periodic,
        unique_pairs=unique_pairs,
        use_numba=use_numba,
        verbose=False,
    )

    n_delta = len(delta_values)

    energy_row = np.zeros((k_energy, n_delta), dtype=float)
    mz_row = np.zeros(n_delta, dtype=float)
    entropy_row = np.zeros(n_delta, dtype=float)
    fidelity_row = np.full(n_delta, np.nan, dtype=float)

    previous_ground_state = None

    for col, delta in enumerate(delta_values):
        chain.J1z = float(delta)

        eigvals, eigvecs = chain.diagonalize(k=k_energy, which="SA")
        ground_state = eigvecs[:, 0]

        energy_row[:, col] = eigvals[:k_energy]
        mz_row[col] = chain.magnetization(ground_state)
        entropy_row[col] = chain.half_chain_entropy(ground_state)

        if previous_ground_state is not None:
            fidelity_row[col] = fidelity(previous_ground_state, ground_state)

        previous_ground_state = ground_state

    return energy_row, mz_row, entropy_row, fidelity_row


def sweep_delta_j2z(
    L: int,
    delta_values: np.ndarray,
    j2z_values: np.ndarray,
    k_energy: int = 5,
    J1xy: float = 1.0,
    J2xy: float = 0.0,
    periodic: bool = True,
    unique_pairs: bool = True,
    use_numba: bool = True,
    n_jobs: int = 1,
) -> Dict[str, np.ndarray]:
    """2D sweep over Delta and J2z."""
    if n_jobs != 1 and not JOBLIB_AVAILABLE:
        warnings.warn("joblib is not available. Falling back to serial execution.")
        n_jobs = 1

    if n_jobs == 1:
        rows = []

        for j2z in tqdm(j2z_values, desc="J2z sweep"):
            rows.append(
                compute_j2z_row(
                    L=L,
                    j2z=float(j2z),
                    delta_values=delta_values,
                    k_energy=k_energy,
                    J1xy=J1xy,
                    J2xy=J2xy,
                    periodic=periodic,
                    unique_pairs=unique_pairs,
                    use_numba=use_numba,
                )
            )

    else:
        rows = Parallel(n_jobs=n_jobs)(
            delayed(compute_j2z_row)(
                L=L,
                j2z=float(j2z),
                delta_values=delta_values,
                k_energy=k_energy,
                J1xy=J1xy,
                J2xy=J2xy,
                periodic=periodic,
                unique_pairs=unique_pairs,
                use_numba=use_numba,
            )
            for j2z in tqdm(j2z_values, desc="J2z sweep")
        )

    n_j2z = len(j2z_values)
    n_delta = len(delta_values)

    energy_surfaces = np.zeros((k_energy, n_j2z, n_delta), dtype=float)
    mz_surface = np.zeros((n_j2z, n_delta), dtype=float)
    entropy_surface = np.zeros((n_j2z, n_delta), dtype=float)
    fidelity_surface = np.full((n_j2z, n_delta), np.nan, dtype=float)

    for row_idx, row_data in enumerate(rows):
        energy_row, mz_row, entropy_row, fidelity_row = row_data

        energy_surfaces[:, row_idx, :] = energy_row
        mz_surface[row_idx, :] = mz_row
        entropy_surface[row_idx, :] = entropy_row
        fidelity_surface[row_idx, :] = fidelity_row

    return {
        "delta_values": delta_values,
        "j2z_values": j2z_values,
        "energy_surfaces": energy_surfaces,
        "mz_surface": mz_surface,
        "entropy_surface": entropy_surface,
        "fidelity_surface": fidelity_surface,
    }


def plot_3d_sweep_results(results: Dict[str, np.ndarray], L: int) -> None:
    """Four-panel 3D surface plot."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is not installed.")

    delta_values = results["delta_values"]
    j2z_values = results["j2z_values"]
    energy_surfaces = results["energy_surfaces"]
    mz_surface = results["mz_surface"]
    entropy_surface = results["entropy_surface"]
    fidelity_surface = results["fidelity_surface"]

    k_energy = energy_surfaces.shape[0]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "surface"}, {"type": "surface"}],
            [{"type": "surface"}, {"type": "surface"}],
        ],
        subplot_titles=(
            "First eigenvalues vs Delta and J2z",
            "Ground-state total Mz vs Delta and J2z",
            "Ground-state fidelity vs Delta and J2z",
            "Ground-state half-chain entanglement entropy vs Delta and J2z",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    energy_colors = ["#2563eb", "#f97316", "#16a34a", "#ef4444", "#8b5cf6"]

    for level in range(k_energy):
        fig.add_trace(
            go.Surface(
                x=delta_values,
                y=j2z_values,
                z=energy_surfaces[level],
                colorscale=[
                    [0, energy_colors[level % len(energy_colors)]],
                    [1, energy_colors[level % len(energy_colors)]],
                ],
                opacity=0.55,
                showscale=False,
                name=f"E{level}",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Surface(
            x=delta_values,
            y=j2z_values,
            z=mz_surface,
            colorscale="Oranges",
            opacity=0.9,
            showscale=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Surface(
            x=delta_values,
            y=j2z_values,
            z=fidelity_surface,
            colorscale="Greens",
            opacity=0.9,
            showscale=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Surface(
            x=delta_values,
            y=j2z_values,
            z=entropy_surface,
            colorscale="Reds",
            opacity=0.9,
            showscale=True,
            colorbar=dict(title="Entropy"),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=f"Sparse periodic XXZ / NNN-XXZ model at L={L}",
        height=1000,
        width=1300,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    for scene_name in ["scene", "scene2", "scene3", "scene4"]:
        fig.layout[scene_name].xaxis.title = "Delta"
        fig.layout[scene_name].yaxis.title = "J2z"
        fig.layout[scene_name].zaxis.title = "Value"

    display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))


# ============================================================
# Entanglement entropy surfaces for increasing L
# ============================================================

def compute_entropy_surface_for_L(
    L: int,
    delta_values: np.ndarray,
    j2z_values: np.ndarray,
    J1xy: float = 1.0,
    J2xy: float = 0.0,
    periodic: bool = True,
    unique_pairs: bool = True,
    use_numba: bool = True,
) -> np.ndarray:
    """Compute ground-state half-chain entropy S(Delta, J2z) for one L."""
    surface = np.zeros((len(j2z_values), len(delta_values)), dtype=float)

    chain = SparseXXZChain(
        L=L,
        J1xy=J1xy,
        J1z=float(delta_values[0]),
        J2xy=J2xy,
        J2z=float(j2z_values[0]),
        periodic=periodic,
        unique_pairs=unique_pairs,
        use_numba=use_numba,
        verbose=True,
    )

    for row, j2z in enumerate(tqdm(j2z_values, desc=f"L={L}, J2z")):
        chain.J2z = float(j2z)

        for col, delta in enumerate(delta_values):
            chain.J1z = float(delta)

            _, ground_state = chain.ground_state()
            surface[row, col] = chain.half_chain_entropy(ground_state)

    return surface


def entropy_surfaces_for_many_L(
    L_values: Sequence[int],
    delta_values: np.ndarray,
    j2z_values: np.ndarray,
    J1xy: float = 1.0,
    J2xy: float = 0.0,
    periodic: bool = True,
    unique_pairs: bool = True,
    use_numba: bool = True,
) -> Dict[int, np.ndarray]:
    """Compute entropy surfaces for multiple L values."""
    surfaces = {}

    for L in L_values:
        print(f"\nComputing entropy surface for L={L}, dim={2**L}")

        surfaces[L] = compute_entropy_surface_for_L(
            L=L,
            delta_values=delta_values,
            j2z_values=j2z_values,
            J1xy=J1xy,
            J2xy=J2xy,
            periodic=periodic,
            unique_pairs=unique_pairs,
            use_numba=use_numba,
        )

    return surfaces


def plot_entropy_surfaces_by_L(
    entropy_surfaces_by_L: Dict[int, np.ndarray],
    delta_values: np.ndarray,
    j2z_values: np.ndarray,
) -> None:
    """Plot one entropy surface per L."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is not installed.")

    L_values = list(entropy_surfaces_by_L.keys())

    n_cols = 2
    n_rows = int(np.ceil(len(L_values) / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[
            [{"type": "surface"} for _ in range(n_cols)]
            for _ in range(n_rows)
        ],
        subplot_titles=[f"L = {L}" for L in L_values],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    for idx, L in enumerate(L_values):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        fig.add_trace(
            go.Surface(
                x=delta_values,
                y=j2z_values,
                z=entropy_surfaces_by_L[L],
                colorscale="Viridis",
                showscale=(idx == len(L_values) - 1),
                colorbar=dict(title="Entropy") if idx == len(L_values) - 1 else None,
                name=f"L={L}",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Ground-state half-chain entanglement entropy vs Delta and J2z",
        height=500 * n_rows,
        width=1200,
        margin=dict(l=10, r=10, t=70, b=10),
    )

    for i in range(1, len(L_values) + 1):
        scene_name = "scene" if i == 1 else f"scene{i}"
        fig.layout[scene_name].xaxis.title = "Delta"
        fig.layout[scene_name].yaxis.title = "J2z"
        fig.layout[scene_name].zaxis.title = "S"

    display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))


# ============================================================
# Optional QuSpin Hamiltonian builder
# ============================================================

def build_quspin_hamiltonian(
    L: int,
    J1xy: float = 1.0,
    J1z: float = 0.0,
    J2xy: float = 0.0,
    J2z: float = 0.0,
    periodic: bool = True,
):
    """
    Optional QuSpin version of the Hamiltonian.

    Requires:
        pip install quspin

    This is especially useful if you want to work in fixed-magnetisation sectors.
    """
    if not QUSPIN_AVAILABLE:
        raise ImportError("QuSpin is not installed. Install with: pip install quspin")

    basis = spin_basis_1d(L, pauli=False)

    nn_pairs = make_pairs(
        L,
        distance=1,
        periodic=periodic,
        unique_pairs=True,
    )

    nnn_pairs = make_pairs(
        L,
        distance=2,
        periodic=periodic,
        unique_pairs=True,
    )

    static = []

    if J1xy != 0.0:
        static.append(["+-", [[J1xy / 2.0, i, j] for i, j in nn_pairs]])
        static.append(["-+", [[J1xy / 2.0, i, j] for i, j in nn_pairs]])

    if J1z != 0.0:
        static.append(["zz", [[J1z, i, j] for i, j in nn_pairs]])

    if J2xy != 0.0:
        static.append(["+-", [[J2xy / 2.0, i, j] for i, j in nnn_pairs]])
        static.append(["-+", [[J2xy / 2.0, i, j] for i, j in nnn_pairs]])

    if J2z != 0.0:
        static.append(["zz", [[J2z, i, j] for i, j in nnn_pairs]])

    H = quspin_hamiltonian(
        static,
        [],
        basis=basis,
        dtype=np.float64,
        check_herm=False,
        check_symm=False,
        check_pcon=False,
    )

    return H, basis


# ============================================================
# Example runners
# ============================================================

def run_santos_2d_example() -> Dict[str, np.ndarray]:
    """Run the original Santos-style Delta sweep for L=6."""
    L = 6
    delta_values = np.linspace(-1.5, 1.5, 61)

    chain = SparseXXZChain(
        L=L,
        J1xy=1.0,
        J1z=delta_values[0],
        J2xy=0.0,
        J2z=0.0,
        periodic=True,
        unique_pairs=True,
        use_numba=True,
        verbose=True,
    )

    results = sweep_delta(
        chain=chain,
        delta_values=delta_values,
        k_energy=5,
        show_progress=True,
    )

    print("computed", len(delta_values), "Delta points for L =", L)
    print(
        "fidelity plotted for",
        len(results["delta_fidelity"]),
        "points on the negative Delta branch",
    )

    plot_santos_2d_results(results=results, L=L)

    return results


def run_delta_j2z_3d_example() -> Dict[str, np.ndarray]:
    """Run a 3D Delta-J2z sweep for L=6."""
    L = 6

    delta_values = np.linspace(-1.5, 1.5, 41)
    j2z_values = np.linspace(-1.5, 1.5, 41)

    results = sweep_delta_j2z(
        L=L,
        delta_values=delta_values,
        j2z_values=j2z_values,
        k_energy=5,
        J1xy=1.0,
        J2xy=0.0,
        periodic=True,
        unique_pairs=True,
        use_numba=True,
        n_jobs=1,
    )

    plot_3d_sweep_results(results=results, L=L)

    return results


def run_entropy_surfaces_example() -> Dict[int, np.ndarray]:
    """Run entropy surfaces for increasing L."""
    L_values = [4, 6, 8, 10]

    delta_values = np.linspace(-1.5, 1.5, 31)
    j2z_values = np.linspace(-1.5, 1.5, 31)

    surfaces = entropy_surfaces_for_many_L(
        L_values=L_values,
        delta_values=delta_values,
        j2z_values=j2z_values,
        J1xy=1.0,
        J2xy=0.0,
        periodic=True,
        unique_pairs=True,
        use_numba=True,
    )

    plot_entropy_surfaces_by_L(
        entropy_surfaces_by_L=surfaces,
        delta_values=delta_values,
        j2z_values=j2z_values,
    )

    return surfaces


if __name__ == "__main__":
    # Uncomment whichever example you want to run.
    #
    # Warning:
    # Exact diagonalisation scales exponentially with L.
    # Start small and increase gradually.

    run_santos_2d_example()

    # run_delta_j2z_3d_example()
    # run_entropy_surfaces_example()
