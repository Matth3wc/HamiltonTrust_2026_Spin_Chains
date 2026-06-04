"""QuSpin-based spin chain tools (scaffold).

Public modules:
- config: configuration dataclasses
- basis: basis builder
- hamiltonian: Hamiltonian construction (NN and NNN XXZ)
- solver: diagonalization utilities
- tracker: state tracking across parameter sweeps
- observables: magnetisation, fidelity, shannon, IPR, r-stat
- sweep: orchestrator to run parameter sweeps
- plotting: simple plotting helpers
"""
from .config import ModelConfig, SweepConfig
from .basis import build_basis
from .hamiltonian import build_hamiltonian
from .solver import Solver
from .tracker import match_states
from .observables import (shannon_entropy, inverse_participation_ratio,
                          fidelity, r_statistic, level_spacings)
                                 entanglement_entropy, magnetisation)

__all__ = [
    "ModelConfig",
    "SweepConfig",
    "build_basis",
    "build_hamiltonian",
    "Solver",
    "match_states",
    "shannon_entropy",
    "inverse_participation_ratio",
    "fidelity",
    "r_statistic",
    "level_spacings",
     "entanglement_entropy",
     "magnetisation",
]
