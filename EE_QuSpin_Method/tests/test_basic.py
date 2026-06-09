import numpy as np
import pytest

# Skip the tests if QuSpin is not available in the current Python environment.
try:
    import quspin  # noqa: F401
except Exception:
    pytest.skip("quspin not installed; skipping EE_QuSpin_Method tests", allow_module_level=True)
from EE_QuSpin_Method.quspin_chain.config import ModelConfig
from EE_QuSpin_Method.quspin_chain.basis import build_basis
from EE_QuSpin_Method.quspin_chain.hamiltonian import build_hamiltonian
from EE_QuSpin_Method.quspin_chain.observables import shannon_entropy, inverse_participation_ratio


def test_build_small_hamiltonian():
    cfg = ModelConfig(L=4, Jxy=1.0, Jz=0.5, pbc=False)
    b = build_basis(cfg)
    H = build_hamiltonian(cfg, b)
    mat = H.toarray()
    assert mat.shape[0] == 2 ** cfg.L


def test_shannon_and_ipr():
    # simple test vector
    v = np.zeros(8, dtype=complex)
    v[0] = 1.0
    s = shannon_entropy(v)
    ipr = inverse_participation_ratio(v)
    assert np.isclose(s, 0.0)
    assert np.isclose(ipr, 1.0)


def test_entanglement_and_magnetisation():
    cfg = ModelConfig(L=4)
    b = build_basis(cfg)
    # all-up product state corresponds to basis.states[0]
    import numpy as np
    vec = np.zeros(b.Ns, dtype=complex)
    vec[0] = 1.0
    # entanglement should be zero for product state
    from EE_QuSpin_Method.quspin_chain.observables import (
        entanglement_entropy,
        magnetisation,
        magnetisation_z,
        magnetisation_z_abs,
        magnetisation_z_squared,
    )
    sent = entanglement_entropy(b, vec, sub_sys_A=list(range(cfg.L // 2)))
    m = magnetisation(b, vec)
    mz = magnetisation_z(b, vec, per_site=False, pauli_units=True)
    mz_abs = magnetisation_z_abs(b, vec, per_site=False, pauli_units=True)
    mz_sq = magnetisation_z_squared(b, vec, per_site=True, pauli_units=True)
    assert np.isclose(sent, 0.0)
    assert np.isclose(m, cfg.L / 2)
    assert np.isclose(mz, cfg.L)
    assert np.isclose(mz_abs, cfg.L)
    assert np.isclose(mz_sq, 1.0)


def test_match_states_uses_optimal_assignment():
    from EE_QuSpin_Method.quspin_chain.tracker import match_states

    prev = np.eye(3, dtype=complex)
    new = prev[:, [1, 2, 0]]
    perm = match_states(prev, new)
    assert perm.tolist() == [2, 0, 1]


def test_run_small_sweep():
    cfg = ModelConfig(L=4, Jxy=1.0)
    from EE_QuSpin_Method.quspin_chain.sweep import run_sweep
    from EE_QuSpin_Method.quspin_chain.config import SweepConfig
    sweep = SweepConfig(param_name='Jz', param_values=[0.1, 0.5, 1.0])
    results = run_sweep(cfg, sweep, k=3, sub_sys_A=list(range(2)))
    assert len(results) == 3
    for r in results:
        assert 'energies' in r and 'states' in r
        assert len(r['energies']) == 3
        assert 'magnetisation_z' in r
        assert 'magnetisation_z_squared' in r
        assert 'tracking_overlap' in r
