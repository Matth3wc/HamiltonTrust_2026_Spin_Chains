"""Orchestrator to run parameter sweeps and collect results."""
from typing import List, Dict, Any
import numpy as np
from .config import ModelConfig, SweepConfig
from .basis import build_basis
from .hamiltonian import build_hamiltonian
from .solver import Solver
from .tracker import match_states
from .observables import (
    shannon_entropy,
    inverse_participation_ratio,
    entanglement_entropy,
    magnetisation,
    fidelity,
    r_statistic,
    level_spacings,
)


def run_sweep(base_cfg: ModelConfig, sweep: SweepConfig, k: int = 5, sub_sys_A=None) -> List[Dict[str, Any]]:
    results = []
    prev_states = None
    prev_states_tracked = None
    for idx, val in enumerate(sweep.param_values):
        # set parameter on the model
        setattr(base_cfg, sweep.param_name, val)
        basis = build_basis(base_cfg)
        H = build_hamiltonian(base_cfg, basis)
        energies, states = Solver.diagonalize(H, k=k)

        # compute level statistics
        ls = level_spacings(energies)
        r = r_statistic(energies)

        # track states between steps by overlap and reorder before computing
        # per-state observables so observables align with the reordered energies
        if prev_states is not None:
            perm = match_states(prev_states, states)
            # reorder states to align with previous step
            states = states[:, perm]
            energies = energies[perm]

        # compute observables for each tracked (and now-ordered) state
        mags = []
        sents = []
        shannons = []
        iprs = []
        fs = []
        for i in range(states.shape[1]):
            vec = states[:, i]
            mags.append(magnetisation(basis, vec))
            sents.append(entanglement_entropy(basis, vec, sub_sys_A=sub_sys_A))
            shannons.append(shannon_entropy(vec))
            iprs.append(inverse_participation_ratio(vec))
            if prev_states_tracked is None:
                fs.append(float('nan'))
            else:
                # fidelity against the previously tracked (ordered) state
                fval = fidelity(prev_states_tracked[:, i], vec)
                fs.append(fval)

        results.append(
            {
                "param": val,
                "energies": energies,
                "states": states,
                "magnetisation": mags,
                "entanglement": sents,
                "shannon": shannons,
                "ipr": iprs,
                "fidelity": fs,
                "level_spacings": ls,
                "r_stat": r,
            }
        )

        prev_states = states.copy()
        prev_states_tracked = states.copy()

    return results
