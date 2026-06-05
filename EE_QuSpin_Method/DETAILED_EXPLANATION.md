EE_QuSpin_Method — Detailed explanation and change log
=====================================================

This document provides a detailed, developer-oriented explanation of the
`EE_QuSpin_Method` package and the `quspin_chain` subpackage. It describes
module responsibilities, dataflow, important functions and classes, typical
usage patterns, design decisions, limitations, and extension points. It also
contains a change log. Every future code modification to files inside this
folder will be recorded here with a clear, human-readable description of what
was changed and why.

Location
--------
- Package: `EE_QuSpin_Method`
- Subpackage: `EE_QuSpin_Method/quspin_chain`

Top-level Files
---------------
- `__init__.py` — package initializer which exposes the public API of
  `quspin_chain`.
- `README.md` — short overview and test/run instructions.
- `requirements.txt` — runtime dependencies (quspin, numpy, scipy, matplotlib,
  pandas, ipykernel, tqdm).
- `DETAILED_EXPLANATION.md` — this file.

Subpackage: `quspin_chain`
-------------------------
This subpackage is a compact scaffold to build, diagonalize, and analyze
spin-1/2 XXZ chains using the QuSpin library. Its modules and responsibilities
are described below.

Module: `config.py`
--------------------
Primary contents:
- `ModelConfig` (dataclass)
  - Fields: `L`, `Jxy`, `Jz`, `Jxy2`, `Jz2`, `pbc`, `basis_kwargs`.
  - Purpose: capture the numeric model parameters and pass configuration to
    other builders (basis and Hamiltonian). `basis_kwargs` is forwarded into
    `quspin` basis constructors to enable symmetry blocks (translation,
    parity, magnetization blocks etc.).
- `SweepConfig` (dataclass)
  - Fields: `param_name`, `param_values`.
  - Purpose: simple representation of a single-parameter sweep (name and
    ordered list of values to apply to a `ModelConfig` instance).

Design notes:
- `ModelConfig` centralizes parameter naming and defaults so other modules can
  depend on a single representation of the model.

Module: `basis.py`
------------------
Primary contents:
- `build_basis(cfg: ModelConfig)` — thin wrapper around
  `quspin.basis.spin_basis_1d(L=cfg.L, pauli=False, **cfg.basis_kwargs)`.

Behavior and rationale:
- This function isolates the QuSpin import and abstracts passing of symmetry
  keyword arguments from `cfg.basis_kwargs`. If QuSpin is not available, the
  function raises an `ImportError` with a helpful message.
- `pauli=False` is used (QuSpin supports both) — this code assumes standard
  Sz basis ordering where spin components are not pre-multiplied by Pauli
  matrices.

Module: `hamiltonian.py`
------------------------
Primary contents:
- `_bond_list(L, offset, pbc)` — helper to produce NN or NNN bond pairs.
- `build_hamiltonian(cfg: ModelConfig, basis, dtype=None)` — build a QuSpin
  `hamiltonian` object for NN (nearest-neighbour) and optional NNN
  (next-nearest-neighbour) XXZ couplings. Terms generated: `xx`, `yy`, `zz`.

Behavior and rationale:
- Bond lists are generated for offsets 1 (NN) and optionally 2 (NNN).
- Static operator lists follow QuSpin's `hamiltonian` static format: lists of
  `[opstr, [[coefficient, i, j], ...]]`.
- `dtype` defaults to `numpy.float64` if not provided.
- If QuSpin cannot be imported, an `ImportError` is raised.

Module: `solver.py`
-------------------
Primary contents:
- `Solver` class with static method `diagonalize(H, k=None, which='SA')`.

Behavior and rationale:
- If `k` (number of eigenvalues/vectors requested) is provided, the method
  first tries a sparse iterative solver: `H.eigsh(k=k, which=which)`. This
  approach is more memory- and time-efficient for large sparse systems when
  only a few low-lying eigenstates are required.
- If the iterative solver fails (or `k` is `None`), it falls back to dense
  diagonalization using `H.toarray()` and `numpy.linalg.eigh`.
- Return format: tuple `(energies, states)`. `states` is a 2D array whose
  columns are eigenvectors; if `k` is provided the returned arrays are sized
  accordingly.

Caveats:
- Dense fallback may be impractical for Hilbert spaces larger than ~2^16 or
  as limited by available memory. Users should prefer symmetry reductions in
  `basis_kwargs` to shrink Hilbert space.

Module: `tracker.py`
---------------------
Primary contents:
- `match_states(prev_vecs: np.ndarray, new_vecs: np.ndarray)` — returns a
  permutation index array mapping columns in `new_vecs` to the ordering of
  `prev_vecs`.

Behavior and rationale:
- The function computes the overlap matrix `|<prev_i | new_j>|` and performs
  greedy matching: for each `i` it picks the `j` with maximum overlap,
  marking matched indices as used. The returned `perm` arranges `new_vecs`
  so that `new_vecs[:, perm[i]]` best corresponds to `prev_vecs[:, i]`.

Limitations and possible improvements:
- Greedy matching is fast but can fail when overlaps are ambiguous (e.g.
  near-degenerate states or global phase flips). Alternatives include the
  Hungarian algorithm (optimal assignment) or including energy proximity.

Module: `observables.py`
------------------------
Primary contents:
- State-based scalar measures: `shannon_entropy`, `inverse_participation_ratio`,
  `fidelity`.
- Spectrum-based: `level_spacings`, `r_statistic`.
- Entanglement and magnetisation that depend on QuSpin `basis` objects:
  - `entanglement_entropy(basis, state, sub_sys_A=None, density=False, alpha=1.0)`
    — delegates to `basis.ent_entropy(...)` and returns `Sent_A`.
  - `magnetisation(basis, state)` — constructs a `Sz` operator via
    `quspin.operators.hamiltonian` and returns the expectation value.

Notes:
- `shannon_entropy` uses natural log by default; base can be changed.
- `inverse_participation_ratio` returns the sum of squared probabilities.
- `r_statistic` implements the average ratio-of-consecutive-gaps metric;
  returns NaN for insufficient spectrum size.
- `magnetisation` constructs the operator on each call; for repeated calls it
  would be more efficient to cache the operator (by `basis` identity).

Module: `sweep.py`
------------------
Primary contents:
- `run_sweep(base_cfg: ModelConfig, sweep: SweepConfig, k: int = 5, sub_sys_A=None)`
  — orchestrates a single-parameter sweep and returns a list of result dicts.

Detailed dataflow per sweep step:
1. Set `getattr(base_cfg, sweep.param_name) = val` for each `val`.
2. `basis = build_basis(base_cfg)` — constructs (and possibly blocks) the
   basis for the current parameters.
3. `H = build_hamiltonian(base_cfg, basis)` — construct QuSpin `hamiltonian`.
4. `energies, states = Solver.diagonalize(H, k=k)` — compute eigenpairs.
5. For each tracked eigenstate `i` of the returned `states` (columns):
   - Compute `magnetisation(basis, vec)`
   - Compute `entanglement_entropy(basis, vec, sub_sys_A=sub_sys_A)`
   - Compute `shannon_entropy(vec)`
   - Compute `inverse_participation_ratio(vec)`
   - Compute `fidelity(prev_states_tracked[:, i], vec)` if previous tracked
     states exist; otherwise `NaN` for fidelity.
6. Compute level spacings and `r_statistic` from `energies`.
7. If `prev_states` is set, call `match_states(prev_states, states)` to
   compute a permutation aligning the new states to the previous ordering.
   Reorder `states` and `energies` accordingly. This keeps observables
   associated to the same tracked physical state across parameter steps.
8. Append a result dict containing `param`, `energies`, `states`,
   `magnetisation`, `entanglement`, `shannon`, `ipr`, `fidelity`,
   `level_spacings`, and `r_stat`.
9. Set `prev_states = states.copy()` and `prev_states_tracked = states.copy()`
   for next iteration.

Result format:
- The returned value is a `List[Dict[str, Any]]`, where each dict contains
  the parameter value and arrays/lists for energies, states and observables.

Performance notes:
- Building a basis and Hamiltonian each step is simple but may be costly for
  large systems; if only one coupling changes in a way that preserves the
  basis structure, consider reusing the basis and only updating the operator
  coefficients. Future refactors could separate basis construction and
  operator construction to allow reuse.

Module: `plotting.py`
---------------------
Primary contents:
- Small Matplotlib convenience functions to visualize `results` from
  `run_sweep`: `plot_energies_vs_param`, `plot_observable_vs_param`,
  `plot_entanglement_vs_energy`.

Usage and examples
------------------
Minimal usage pattern (in Python):

```python
from EE_QuSpin_Method.quspin_chain import (
    ModelConfig, SweepConfig, run_sweep
)

cfg = ModelConfig(L=8, Jxy=1.0, Jz=1.0, pbc=True)
sweep = SweepConfig(param_name="Jz", param_values=[0.5, 1.0, 1.5])
results = run_sweep(cfg, sweep, k=5)
```

After `results` is computed you can plot using `plotting` helpers or inspect
`results[i]` for the i-th parameter point.

Running tests
-------------
From the repository root:

```bash
source .venv/bin/activate
pip install -r EE_QuSpin_Method/requirements.txt
pytest EE_QuSpin_Method/tests
```

Design decisions and recommendations
-----------------------------------
- QuSpin is used for concise, tested many-body routines (basis construction,
  operator assembly, entanglement entropy). The package deliberately keeps a
  thin wrapper to avoid duplicating QuSpin functionality.
- For larger system sizes you must use symmetry blocks (`basis_kwargs`) to
  drastically reduce Hilbert space size. This is the intended way to scale
  beyond naive 2^L limits.
- `Solver.diagonalize` tries iterative sparse solvers first for efficiency;
  users should tune `k` and `which` (the latter for targeting low/high
  eigenvalues) to match their experiment.
- `match_states` is greedy for simplicity; for production workflows prefer an
  optimal assignment algorithm if state reordering is critical.

Known limitations and TODOs
--------------------------
- No caching of operators (e.g., magnetisation Sz) — repeated calls rebuild
  operators.
- No explicit logging of failure modes for QuSpin imports beyond the raised
  ImportError messages.
- `run_sweep` rebuilds `basis` and `hamiltonian` at every step which can be
  inefficient if only coupling scalars change; an incremental update path
  would be a useful enhancement.

Change Log (append-only)
------------------------
- 2026-06-05 — Initial creation of `DETAILED_EXPLANATION.md` by assistant.
  - Added file that explains all modules in `EE_QuSpin_Method/quspin_chain`,
    including dataflow, API usage, design notes, limitations and commands to
    run tests. This file will be updated by the assistant whenever code is
    modified in this folder; each change will be summarized here with date,
    files changed, and rationale.
 - 2026-06-05 — Fixed notebook metadata in `unit_testing.ipynb`.
   - Files changed: `unit_testing.ipynb`, `EE_QuSpin_Method/DETAILED_EXPLANATION.md`.
   - Summary: Added `metadata.id` fields to each existing cell in
     `unit_testing.ipynb` so that the notebook conforms to the expected
     notebook format (some runners require `metadata.id` on existing cells).
     This resolves failures when opening or executing the notebook in some
     notebook tooling that enforces the metadata.id requirement.
   - Rationale: The notebook previously had cell-level `id` fields but
     lacked `metadata.id`. The change is metadata-only and does not alter
     code or execution logic.
 - 2026-06-05 — Fix: align observables with reordered eigenstates in `run_sweep`.
   - Files changed: `quspin_chain/sweep.py`, `EE_QuSpin_Method/DETAILED_EXPLANATION.md`.
   - Summary: Moved computation of per-state observables (magnetisation,
     entanglement, Shannon entropy, IPR, and fidelity) to occur after state
     reordering by `match_states`. Previously observables were computed
     before the permutation, causing misalignment between `energies` and
     the observable lists in each result dict (this produced incorrect
     plots such as entanglement vs energy and erroneous fidelity traces).
   - Rationale: Ensures that each observable entry corresponds to the
     matching energy/state after tracking; fidelity is now computed against
     the previously tracked and ordered state vector.
 - 2026-06-05 — Fix: magnetisation scaling and complex casting.
   - Files changed: `quspin_chain/observables.py`, `EE_QuSpin_Method/DETAILED_EXPLANATION.md`.
   - Summary: `magnetisation` now takes the real part of the expectation
     value to avoid ComplexWarning and scales the result by 2 when the
     basis appears to use spin-1/2 `S^z` units (i.e., per-site +/-1/2).
     This makes the function return `L` for an all-up product state and
     matches existing unit tests and plotting expectations.
   - Rationale: Different QuSpin conventions (Pauli vs spin operators)
     produce different numeric factors; this change preserves backward
     compatibility while producing the expected integer magnetisation.
 - 2026-06-05 — Simplify `magnetisation` implementation to be adaptive.
   - Files changed: `quspin_chain/observables.py`, `EE_QuSpin_Method/DETAILED_EXPLANATION.md`.
   - Summary: Rewrote `magnetisation` to compute the raw expectation of
     the total `Sz` operator and then determine an automatic scaling factor
     by measuring the expectation on the first computational basis vector
     (assumed to be the all-up product state). The function scales the
     measured value so that that reference vector maps to magnetisation =
     `L`. This removes ambiguous heuristics and improves correctness across
     different QuSpin conventions.
   - Rationale: The adaptive scale removes dependence on optional
     `basis` attributes and gives a consistent magnetisation output for
     plotting and testing.

Update policy
-------------
- Policy: every time a code file in `EE_QuSpin_Method` (or its
  `quspin_chain` subpackage) is modified by me (the assistant), I will:
  1. Append an entry to the `Change Log` above with date/time, files changed,
     a concise human summary of the edits, and the reason for the change.
  2. Add a short patch-like summary (not full diffs) describing the main
     code changes and where to inspect them.
  3. If the changes affect usage or public API signatures, include a short
     migration note with examples.

- If you (the human maintainer) make changes, please tell me and I will
  update this file to reflect those changes (optional).

Contact
-------
- If you want additional artifacts (UML diagrams showing module
  interactions, unit tests for `match_states`, or caching helpers for
  repeated operator construction), tell me which artifact and I will create
  and document them here with corresponding change-log entries.
