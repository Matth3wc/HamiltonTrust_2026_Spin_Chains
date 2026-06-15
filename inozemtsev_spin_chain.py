"""
Inozemtsev spin chain in the full 2^L Hilbert space.

This file replaces the nearest-neighbour / next-nearest-neighbour XXZ Hamiltonian
with the all-to-all isotropic Inozemtsev Hamiltonian

    H = sum_{i < j} J_ij * a * (1 - sigma_i . sigma_j)

where

    J_ij = WeierstrassP(i - j)

is evaluated numerically using a truncated lattice sum.  The default value
`operator_prefactor = 0.5` gives the common convention

    H = sum_{i < j} J_ij * (1 - sigma_i . sigma_j) / 2.

If you want the no-half convention instead, set `operator_prefactor = 1.0`.

The code keeps the same basic workflow as the original XXZ file:
    - build the dense Hamiltonian in the computational basis,
    - diagonalize it,
    - compute ground-state magnetisation,
    - compute half-chain entanglement entropy,
    - compute fidelity between neighbouring parameter points,
    - plot the first few energies and ground-state observables.

Warning:
    This uses a dense 2^L x 2^L Hamiltonian.  It is fine for small L, but the
    memory cost scales as 4^L.  Be careful beyond L ~ 12 unless you move to
    sparse matrices and/or symmetry sectors.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass


# -----------------------------------------------------------------------------
# Weierstrass elliptic coupling
# -----------------------------------------------------------------------------

def weierstrass_p_lattice(z: float | complex, L: int, kappa: float = 1.0, cutoff: int = 8) -> float:
    """Approximate the Weierstrass elliptic function by a truncated lattice sum.

    The chain period is taken to be L and the second period is chosen as

        omega_2 = i*pi/kappa.

    So kappa controls the interaction range.  This is a numerical replacement
    for Mathematica's WeierstrassP.

    Parameters
    ----------
    z:
        Pair separation i - j.  In the spin chain this is a non-zero integer.
    L:
        Chain length, used as the real period.
    kappa:
        Elliptic/range parameter.  Larger kappa makes the imaginary period
        shorter.  Sweep this parameter to change the long-range profile.
    cutoff:
        Number of lattice images kept in each direction.  Increasing this gives
        a more accurate but slower approximation.

    Returns
    -------
    float
        Real part of the truncated WeierstrassP value.
    """
    if L <= 1:
        raise ValueError("L must be at least 2.")
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    if cutoff < 1:
        raise ValueError("cutoff must be at least 1.")

    z = complex(z)
    period_1 = complex(L)
    period_2 = 1j * np.pi / float(kappa)

    if abs(z) < 1e-14:
        raise ValueError("WeierstrassP is singular at z = 0. Use only i != j pairs.")

    total = 1.0 / (z * z)

    for m in range(-cutoff, cutoff + 1):
        for n in range(-cutoff, cutoff + 1):
            if m == 0 and n == 0:
                continue

            omega = m * period_1 + n * period_2
            total += 1.0 / (z - omega) ** 2 - 1.0 / omega**2

    # For the physical integer separations used here, the exact answer is real.
    # The truncated sum can leave a tiny imaginary numerical residue.
    return float(np.real(total))


# -----------------------------------------------------------------------------
# Main chain class
# -----------------------------------------------------------------------------

class FullInozemtsevChain:
    """Periodic Inozemtsev spin chain in the full 2^L Hilbert space.

    Computational basis convention:
        site 0 is the leftmost/highest bit,
        bit 1 means spin up,
        bit 0 means spin down.

    Pair operator:
        H_ij = J_ij * a * (1 - sigma_i . sigma_j),

    where `a = operator_prefactor`.

    In the two-spin basis this means:
        |up up>       -> 0,
        |down down>   -> 0,
        |up down>     -> 2*a*J_ij |up down> - 2*a*J_ij |down up>,
        |down up>     -> 2*a*J_ij |down up> - 2*a*J_ij |up down>.

    With the default a = 0.5, opposite-spin pairs contribute
        diagonal += J_ij,
        flipped  += -J_ij.
    """

    def __init__(
        self,
        L: int,
        kappa: float = 1.0,
        cutoff: int = 8,
        normalize_couplings: bool = True,
        operator_prefactor: float = 0.5,
    ):
        self.L = int(L)
        self.kappa = float(kappa)
        self.cutoff = int(cutoff)
        self.normalize_couplings = bool(normalize_couplings)
        self.operator_prefactor = float(operator_prefactor)

        if self.L <= 1:
            raise ValueError("L must be at least 2.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")

        self.dim = 1 << self.L
        self.states = np.arange(self.dim, dtype=np.uint64)

        # Inozemtsev is all-to-all: every unordered pair i < j appears.
        self.pairs = [(i, j) for i in range(self.L) for j in range(i + 1, self.L)]

        # Precompute M_z values for fast magnetisation expectation values.
        # This is total S^z in units where spin up contributes +1/2 and spin
        # down contributes -1/2.
        self.mz_values = np.array(
            [int(state).bit_count() - self.L / 2 for state in self.states],
            dtype=float,
        )

        self.couplings = self._compute_couplings()

    def _compute_couplings(self) -> dict[tuple[int, int], float]:
        """Compute all pair couplings J_ij = WeierstrassP(i - j)."""
        couplings: dict[tuple[int, int], float] = {}

        for i, j in self.pairs:
            couplings[(i, j)] = weierstrass_p_lattice(
                i - j,
                L=self.L,
                kappa=self.kappa,
                cutoff=self.cutoff,
            )

        if self.normalize_couplings:
            # Normalising by the nearest-neighbour coupling makes sweeps over
            # kappa easier to interpret: kappa changes the relative range of the
            # interactions rather than merely changing the total energy scale.
            nn_coupling = weierstrass_p_lattice(
                1,
                L=self.L,
                kappa=self.kappa,
                cutoff=self.cutoff,
            )
            if abs(nn_coupling) < 1e-14:
                raise ZeroDivisionError("Nearest-neighbour Weierstrass coupling is too close to zero.")
            couplings = {pair: value / nn_coupling for pair, value in couplings.items()}

        return couplings

    def update_kappa(self, kappa: float) -> None:
        """Update kappa and recompute all pair couplings."""
        self.kappa = float(kappa)
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        self.couplings = self._compute_couplings()

    def build_hamiltonian(self) -> np.ndarray:
        """Build the dense Inozemtsev Hamiltonian matrix."""
        H = np.zeros((self.dim, self.dim), dtype=float)

        for row, state in enumerate(self.states):
            state_int = int(state)
            diagonal = 0.0

            for left, right in self.pairs:
                Jij = self.couplings[(left, right)]

                bit_left = self.L - 1 - left
                bit_right = self.L - 1 - right

                occ_left = (state_int >> bit_left) & 1
                occ_right = (state_int >> bit_right) & 1

                # a * (1 - sigma.sigma) gives a contribution only when the two
                # spins are opposite.  The strength below is 2*a*Jij.
                if occ_left != occ_right:
                    pair_strength = 2.0 * self.operator_prefactor * Jij
                    diagonal += pair_strength

                    flipped = state_int ^ ((1 << bit_left) | (1 << bit_right))
                    H[row, flipped] += -pair_strength

            H[row, row] += diagonal

        return H

    def diagonalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all eigenvalues and eigenvectors of the dense Hamiltonian."""
        H = self.build_hamiltonian()
        eigvals, eigvecs = np.linalg.eigh(H)
        return eigvals, eigvecs

    def magnetization(self, state: np.ndarray) -> float:
        """Total M_z expectation value for a state vector."""
        probabilities = np.abs(state) ** 2
        return float(np.dot(probabilities, self.mz_values))

    def half_chain_entropy(self, state: np.ndarray) -> float:
        """Half-chain von Neumann entanglement entropy."""
        ell = self.L // 2
        psi_matrix = state.reshape((1 << ell, 1 << (self.L - ell)))
        singular_values = np.linalg.svd(psi_matrix, compute_uv=False)
        probabilities = singular_values**2
        probabilities = probabilities[probabilities > 1e-14]
        return float(-np.sum(probabilities * np.log(probabilities)))

    def coupling_table(self) -> list[tuple[int, int, int, float]]:
        """Return a readable list of (i, j, periodic_distance, J_ij)."""
        rows = []
        for i, j in self.pairs:
            raw_distance = abs(i - j)
            periodic_distance = min(raw_distance, self.L - raw_distance)
            rows.append((i, j, periodic_distance, self.couplings[(i, j)]))
        return rows


# -----------------------------------------------------------------------------
# Observables and sweep helpers
# -----------------------------------------------------------------------------

def fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Absolute overlap between two state vectors."""
    return float(np.abs(np.vdot(state_a, state_b)))


def run_kappa_sweep(
    L: int = 6,
    kappa_values: np.ndarray | None = None,
    cutoff: int = 8,
    normalize_couplings: bool = True,
    operator_prefactor: float = 0.5,
    levels_to_keep: int = 5,
) -> dict[str, np.ndarray]:
    """Sweep kappa and compute energies plus ground-state observables."""
    if kappa_values is None:
        kappa_values = np.linspace(0.2, 3.0, 57)

    kappa_values = np.asarray(kappa_values, dtype=float)

    energy_levels = []
    mz_values = []
    entropy_values = []
    ground_states = []

    chain = FullInozemtsevChain(
        L=L,
        kappa=float(kappa_values[0]),
        cutoff=cutoff,
        normalize_couplings=normalize_couplings,
        operator_prefactor=operator_prefactor,
    )

    for kappa in kappa_values:
        chain.update_kappa(float(kappa))
        eigvals, eigvecs = chain.diagonalize()

        ground_state = eigvecs[:, 0]
        energy_levels.append(eigvals[:levels_to_keep])
        mz_values.append(chain.magnetization(ground_state))
        entropy_values.append(chain.half_chain_entropy(ground_state))
        ground_states.append(ground_state)

    energy_levels = np.asarray(energy_levels, dtype=float)
    mz_values = np.asarray(mz_values, dtype=float)
    entropy_values = np.asarray(entropy_values, dtype=float)
    ground_states = np.asarray(ground_states, dtype=complex)

    fidelity_values = np.full(len(kappa_values), np.nan, dtype=float)
    for idx in range(1, len(kappa_values)):
        fidelity_values[idx] = fidelity(ground_states[idx - 1], ground_states[idx])

    return {
        "L": np.array(L),
        "kappa_values": kappa_values,
        "energy_levels": energy_levels,
        "mz_values": mz_values,
        "entropy_values": entropy_values,
        "fidelity_values": fidelity_values,
        "ground_states": ground_states,
    }


def plot_kappa_sweep(results: dict[str, np.ndarray]) -> None:
    """Make the four-panel plot analogous to the original XXZ Delta plot."""
    L = int(results["L"])
    kappa_values = results["kappa_values"]
    energy_levels = results["energy_levels"]
    mz_values = results["mz_values"]
    entropy_values = results["entropy_values"]
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
            kappa_values,
            energy_levels[:, level],
            label=f"E{level}",
            linewidth=2,
            color=level_colors[level % len(level_colors)],
        )
    axes[0, 0].set_title("First eigenvalues vs kappa", color=panel_color)
    axes[0, 0].set_ylabel("Energy")
    axes[0, 0].legend(ncol=3, frameon=False)
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].set_facecolor("#f8fafc")

    axes[0, 1].plot(kappa_values, mz_values, color=accent_mz, linewidth=2)
    axes[0, 1].set_title("Ground-state total $M_z$ vs kappa", color=panel_color)
    axes[0, 1].set_ylabel("$M_z$")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].set_facecolor("#f8fafc")

    axes[1, 0].plot(kappa_values, fidelity_values, color=accent_fidelity, linewidth=2)
    axes[1, 0].set_title("Ground-state fidelity vs kappa", color=panel_color)
    axes[1, 0].set_xlabel("kappa")
    axes[1, 0].set_ylabel("Fidelity")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].set_facecolor("#f8fafc")

    axes[1, 1].plot(kappa_values, entropy_values, color=accent_entropy, linewidth=2)
    axes[1, 1].set_title("Ground-state half-chain entropy vs kappa", color=panel_color)
    axes[1, 1].set_xlabel("kappa")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].set_facecolor("#f8fafc")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(
        f"Periodic Inozemtsev spin chain at L={L} in the full $2^L$ Hilbert space",
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def compute_entropy_curve_for_L_values(
    L_values: list[int],
    kappa_values: np.ndarray,
    cutoff: int = 8,
    normalize_couplings: bool = True,
    operator_prefactor: float = 0.5,
) -> dict[int, np.ndarray]:
    """Compute ground-state half-chain entropy versus kappa for several L."""
    entropy_by_L: dict[int, np.ndarray] = {}

    for L in L_values:
        print(f"Computing entropy curve for L={L}, Hilbert dimension={2**L}")
        results = run_kappa_sweep(
            L=L,
            kappa_values=kappa_values,
            cutoff=cutoff,
            normalize_couplings=normalize_couplings,
            operator_prefactor=operator_prefactor,
            levels_to_keep=1,
        )
        entropy_by_L[int(L)] = results["entropy_values"]

    return entropy_by_L


def plot_entropy_curves_for_L_values(kappa_values: np.ndarray, entropy_by_L: dict[int, np.ndarray]) -> None:
    """Plot S(kappa) for several chain lengths."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for L, entropy_values in entropy_by_L.items():
        ax.plot(kappa_values, entropy_values, linewidth=2, label=f"L={L}")

    ax.set_title("Ground-state half-chain entropy vs kappa for increasing L")
    ax.set_xlabel("kappa")
    ax.set_ylabel("Entropy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()


# -----------------------------------------------------------------------------
# Example run
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Main single-L sweep, replacing the old Delta sweep.
    L = 6
    kappa_values = np.linspace(0.2, 3.0, 57)

    results = run_kappa_sweep(
        L=L,
        kappa_values=kappa_values,
        cutoff=8,
        normalize_couplings=True,
        operator_prefactor=0.5,
        levels_to_keep=5,
    )

    print(f"Computed {len(kappa_values)} kappa points for L={L}")
    plot_kappa_sweep(results)

    # Increasing-L entropy curves.  Keep this modest because the code uses full
    # dense diagonalisation.  Increase only carefully.
    L_values = [4, 6, 8]
    kappa_values_entropy = np.linspace(0.2, 3.0, 29)

    entropy_by_L = compute_entropy_curve_for_L_values(
        L_values=L_values,
        kappa_values=kappa_values_entropy,
        cutoff=8,
        normalize_couplings=True,
        operator_prefactor=0.5,
    )

    plot_entropy_curves_for_L_values(kappa_values_entropy, entropy_by_L)

    plt.show()
