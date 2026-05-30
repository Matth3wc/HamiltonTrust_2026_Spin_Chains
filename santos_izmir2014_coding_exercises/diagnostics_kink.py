import numpy as np
from santos_izmir2014_coding_exercises.hybrid_max_scale import ExactHalfFillingXXZ
from scipy.sparse.linalg import eigsh


def rotate_state_uint64(s, L):
    # s is numpy.uint64 or int
    new = np.uint64(0)
    for site in range(L):
        bitpos = L - 1 - site
        if (int(s) >> bitpos) & 1:
            newpos = (site + 1) % L
            newbit = 1 << (L - 1 - newpos)
            new |= np.uint64(newbit)
    return new


def diagnostics_for_L(L, j2_min=0.6, j2_max=1.3, j2_step=0.02):
    print(f"# Diagnostics for L={L}")
    Nup = L // 2
    chain = ExactHalfFillingXXZ(L=L, Nup=Nup, periodic=True)
    j2_values = np.round(np.arange(j2_min, j2_max + 1e-12, j2_step), 12)

    # build permutation for translation-by-1
    dim = chain.hilbert_dimension()
    perm = np.empty(dim, dtype=np.int32)
    for i, s in enumerate(chain.states):
        r = rotate_state_uint64(s, L)
        perm[i] = int(chain.index_map[int(r)])

    psi_prev = None
    for j2 in j2_values:
        H = chain.hamiltonian(j2xy=float(j2), j2z=0.0)
        # compute lowest two eigenvalues and vectors for gap
        if H.shape[0] == 1:
            E0 = float(H[0, 0])
            gap = 0.0
            psi = np.array([1.0], dtype=float)
        else:
            try:
                vals, vecs = eigsh(H, k=2, which='SA', tol=1e-10, maxiter=200000, v0=psi_prev)
            except Exception:
                # fallback to k=1 then estimate gap separately
                vals1, vec1 = eigsh(H, k=1, which='SA', tol=1e-10, maxiter=200000)
                E0 = float(np.min(vals1))
                psi = vec1[:, 0]
                # compute second eigenvalue with shift-invert could be expensive; set NaN
                gap = np.nan
            else:
                idx = np.argsort(vals)
                vals = vals[idx]
                vecs = vecs[:, idx]
                E0 = float(vals[0])
                E1 = float(vals[1])
                gap = float(E1 - E0)
                psi = vecs[:, 0]

        # fidelity with previous ground state
        fidelity = np.nan
        if psi_prev is not None:
            fidelity = float(abs(np.vdot(psi_prev, psi)))

        # translation expectation
        t_psi = psi[perm]
        trans = np.vdot(psi.conj(), t_psi)
        trans_mag = float(abs(trans))
        trans_phase = float(np.angle(trans))

        # dimer (alternating bond energy) using nearest-neighbour bond operators
        bond_energies = []
        for bi, (a, b) in enumerate(chain.nn_pairs):
            # build single-bond operator
            Hbond = chain._term_matrix(np.asarray([[a, b]], dtype=np.int32), chain.J1xy, chain.J1z)
            be = float(np.vdot(psi.conj(), Hbond.dot(psi)))
            bond_energies.append(be)
        bond_energies = np.asarray(bond_energies, dtype=float)
        # alternating sum normalized
        alt = np.sum(((-1) ** np.arange(L)) * bond_energies) / L

        print(f"{L} {j2:.6f} {E0:.12e} {gap:.12e} {fidelity:.12e} {trans_mag:.12e} {trans_phase:.12e} {alt:.12e}")

        psi_prev = psi


if __name__ == '__main__':
    L_list = [8, 10, 12, 14, 16]
    for L in L_list:
        diagnostics_for_L(L, j2_min=0.6, j2_max=1.3, j2_step=0.02)
