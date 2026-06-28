#!/usr/bin/env python
"""Manifold Probe + superposition metrics for the SM-ELF code space.

The Manifold Probe is from "Probing for Representation Manifolds in Superposition"
(arXiv:2605.18537). It generalizes a linear regression probe: for a concept with a
scalar label z, it learns smooth features f_k(z)=beta_k^T h(z) that are *linearly
predictable* from a representation X, together with the encoding directions u_k such
that X ~= sum_k u_k f_k(z) + c. We apply it with X = the SM-ELF code c in R^k and
z = an attribute label (sentiment / gender / animal / length).

Faithful core (their Def. 3 / Prop. 4), per concept, sequentially over features:
    min_{f,w,b}  sum_i (f(z_i) - w^T x_i - b)^2 + lam_w ||w||^2 + lam_f J(f)
    s.t.  sum_i f(z_i)=0,  (1/n) sum_i f(z_i)^2 = 1,  f _|_ f_1..f_{k-1}
with f(z)=beta^T h(z), J(f)=beta^T S beta (integrated squared 2nd-derivative).
Closed form (Prop. 4): generalized eigenproblem  M beta = nu Sigma beta,
    A = X (X^T X + lam_w I)^-1 X^T   (ridge hat matrix)
    M = H^T (I - A) H + lam_f S,   Sigma = (1/n) H^T H
features ranked by SMALLEST nu (smallest residual = most linearly encoded), and
encoding direction  u_k = (1/n) X^T H beta_k.

Superposition metrics (this module's construction on top of the recovered
directions, grounded in Elhage et al., "Toy Models of Superposition": more concept
features than dimensions => features share non-orthogonal directions). We report,
per model: each concept's manifold dimension (# features with held-out R^2 above a
threshold) and best R^2; the cross-concept interference matrix (|cos| / principal-
angle overlap of encoding subspaces); the mean off-diagonal interference; the
random-direction baseline E|cos|~sqrt(2/(pi k)); and a capacity ratio
(sum of concept manifold dims)/k. Higher off-diagonal interference *above the
baseline*, and capacity ratio > 1, indicate attributes packed into superposition.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Smooth polynomial basis over a scalar label z, with a 2nd-derivative penalty.
# --------------------------------------------------------------------------- #
def _poly_basis(z, degree):
    """Return H = [1, z, z^2, ..., z^degree] evaluated at z (n,) -> (n, degree+1)."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    return np.vander(z, N=degree + 1, increasing=True)  # columns z^0..z^degree


def _smoothness_penalty(degree, zmin, zmax, grid=512):
    """S_{ij} = ∫ h_i''(z) h_j''(z) dz over [zmin, zmax] for the monomial basis.

    h_j(z)=z^j -> h_j''(z)=j(j-1) z^{j-2}. Integrated numerically (trapezoid).
    """
    zg = np.linspace(zmin, zmax, grid)
    # second derivative of each monomial column on the grid
    D2 = np.zeros((grid, degree + 1))
    for j in range(degree + 1):
        if j >= 2:
            D2[:, j] = j * (j - 1) * zg ** (j - 2)
    return np.trapz(D2[:, :, None] * D2[:, None, :], zg, axis=0)  # (q, q)


def _standardize_label(z):
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    lo, hi = z.min(), z.max()
    if hi - lo < 1e-12:
        return np.zeros_like(z), 0.0, 1.0
    zs = 2.0 * (z - lo) / (hi - lo) - 1.0  # -> [-1, 1]
    return zs, lo, hi


def fit_manifold_probe(X, z, degree=None, lam_w=1e-2, lam_f=1e-3,
                       test_frac=0.3, r2_threshold=0.05, seed=0, max_features=6):
    """Fit the Manifold Probe of concept z onto representations X.

    Args:
      X: (n, p) representations (e.g. codes c in R^k), will be centered.
      z: (n,) scalar concept label.
      degree: polynomial-basis degree; default min(5, #unique(z)-1), >=1.
    Returns dict with: features f_k (train), encoding directions U (p, K)
      (unit-norm columns), held-out test R^2 per feature, manifold_dim, r2_max.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    zs, _, _ = _standardize_label(z)
    n_unique = len(np.unique(np.round(zs, 6)))
    if degree is None:
        degree = int(min(5, max(1, n_unique - 1)))
    degree = max(1, min(degree, n_unique - 1)) if n_unique > 1 else 1

    # train/test split (for honest held-out R^2 of each recovered feature)
    idx = rng.permutation(n)
    n_te = max(2, int(round(test_frac * n)))
    te, tr = idx[:n_te], idx[n_te:]

    Xtr = X[tr] - X[tr].mean(0, keepdims=True)
    Xte = X[te] - X[tr].mean(0, keepdims=True)
    Htr = _poly_basis(zs[tr], degree)
    Htr = Htr - Htr.mean(0, keepdims=True)        # center features (sum f = 0)
    Hte = _poly_basis(zs[te], degree) - _poly_basis(zs[tr], degree).mean(0, keepdims=True)
    q = Htr.shape[1]
    ntr = Xtr.shape[0]

    S = _smoothness_penalty(degree, -1.0, 1.0)
    # ridge hat matrix A = X (X^T X + lam_w I)^-1 X^T  (ntr x ntr)
    XtX = Xtr.T @ Xtr + lam_w * np.eye(p)
    A = Xtr @ np.linalg.solve(XtX, Xtr.T)
    M = Htr.T @ (np.eye(ntr) - A) @ Htr + lam_f * S          # (q, q)
    Sigma = (Htr.T @ Htr) / ntr + 1e-8 * np.eye(q)           # (q, q), unit-var constraint

    # generalized eigenproblem M beta = nu Sigma beta  via Cholesky whitening
    L = np.linalg.cholesky(Sigma)
    Linv = np.linalg.inv(L)
    Mt = Linv @ M @ Linv.T
    Mt = 0.5 * (Mt + Mt.T)
    nu, W = np.linalg.eigh(Mt)            # ascending eigenvalues
    Betas = Linv.T @ W                    # back to beta space; columns are features
    order = np.argsort(nu)               # smallest residual first

    feats_r2, U_cols = [], []
    K = min(max_features, q)
    for j in order[:K]:
        beta = Betas[:, j]
        # normalize feature to unit train variance
        ftr = Htr @ beta
        s = np.sqrt(np.mean(ftr ** 2)) + 1e-12
        beta = beta / s
        ftr = Htr @ beta
        fte = Hte @ beta
        # held-out R^2 of predicting this feature linearly from X (ridge); w is the
        # READOUT direction. We use w (not the covariance encoding direction
        # X^T f) because superposition that causes off-target *leakage* is
        # interference between linear read-outs, and our steering axis is itself a
        # linear read-out (difference-of-means). The covariance direction X^T f is
        # dominated by the code's high-variance mode in anisotropic spaces and is
        # not the leakage-relevant geometry.
        w = np.linalg.solve(XtX, Xtr.T @ ftr)
        un = w / (np.linalg.norm(w) + 1e-12)
        pred = Xte @ w
        ss_res = np.sum((fte - pred) ** 2)
        ss_tot = np.sum((fte - fte.mean()) ** 2) + 1e-12
        feats_r2.append(1.0 - ss_res / ss_tot)
        U_cols.append(un)

    feats_r2 = np.array(feats_r2)
    U = np.stack(U_cols, axis=1) if U_cols else np.zeros((p, 0))
    # order recovered features by held-out R^2 (most-encoded first)
    o2 = np.argsort(-feats_r2)
    feats_r2, U = feats_r2[o2], U[:, o2]

    # The concept's manifold dimension is the rank of the span of its *encoding
    # directions* (R^2-weighted), NOT the count of linearly-predictable features:
    # several smooth features can re-use the same direction (e.g. a purely linear
    # concept yields one direction however many polynomial features fit it). SVD the
    # thresholded, R^2-weighted directions and count significant singular values.
    keep = feats_r2 >= r2_threshold
    if np.any(keep):
        Uw = U[:, keep] * np.sqrt(np.clip(feats_r2[keep], 0, None))[None, :]
        Wsvd, svals, _ = np.linalg.svd(Uw, full_matrices=False)
        manifold_dim = int(np.sum(svals >= 0.1 * (svals[0] + 1e-12)))
        manifold_dim = max(1, manifold_dim)
        Q = Wsvd[:, :manifold_dim]            # orthonormal basis of encoding subspace
    else:
        manifold_dim, Q = 0, np.zeros((p, 0))
    return {
        "test_r2": feats_r2,
        "U": U,                              # (p, K) raw unit-norm encoding directions
        "Q": Q,                              # (p, manifold_dim) orthonormal subspace basis
        "manifold_dim": manifold_dim,
        "r2_max": float(feats_r2.max()) if len(feats_r2) else 0.0,
        "degree": degree,
    }


def _subspace_overlap(Ua, Ub):
    """Mean squared cosine of principal angles between span(Ua), span(Ub).

    For 1-D subspaces this is cos^2 of the angle between the two directions.
    Returns a value in [0,1]; 0 = orthogonal subspaces, 1 = identical.
    """
    if Ua.shape[1] == 0 or Ub.shape[1] == 0:
        return 0.0
    Qa, _ = np.linalg.qr(Ua)
    Qb, _ = np.linalg.qr(Ub)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.mean(s ** 2))


def superposition_report(X, labels, dim, use_subspace=True, **probe_kw):
    """Run the Manifold Probe for each named concept and compute superposition metrics.

    Args:
      X: (n, p) codes.   labels: dict name -> (n,) scalar label.   dim: code dim k.
    Returns dict: per-concept probe summary, interference matrix, mean off-diagonal
      interference, random baseline, and capacity ratio.
    """
    names = list(labels.keys())
    probes = {}
    for name in names:
        z = np.asarray(labels[name], dtype=np.float64)
        # restrict to examples with a defined (non-zero / finite) label where relevant
        probes[name] = fit_manifold_probe(X, z, **probe_kw)

    m = len(names)
    inter = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if use_subspace:
                # overlap of the two concepts' orthonormal encoding subspaces Q
                inter[i, j] = _subspace_overlap(probes[names[i]]["Q"], probes[names[j]]["Q"])
            else:
                Ui, Uj = probes[names[i]]["U"], probes[names[j]]["U"]
                ui = Ui[:, :1] if Ui.shape[1] else np.zeros((X.shape[1], 1))
                uj = Uj[:, :1] if Uj.shape[1] else np.zeros((X.shape[1], 1))
                inter[i, j] = float((ui[:, 0] @ uj[:, 0]) ** 2)

    off = inter[~np.eye(m, dtype=bool)]
    mean_off = float(off.mean()) if off.size else 0.0
    # random baseline: E[cos^2] of two random unit vectors in R^k is 1/k
    baseline = 1.0 / max(1, dim)
    total_mdim = sum(probes[n]["manifold_dim"] for n in names)
    return {
        "dim": int(dim),
        "concepts": names,
        "probes": {n: {"manifold_dim": probes[n]["manifold_dim"],
                       "r2_max": probes[n]["r2_max"],
                       "test_r2": probes[n]["test_r2"].round(3).tolist()} for n in names},
        "interference": inter,                 # (m, m) subspace overlap in [0,1]
        "mean_offdiag_interference": mean_off,
        "random_baseline_cos2": baseline,
        "excess_interference": mean_off - baseline,
        "capacity_ratio": total_mdim / max(1, dim),
        "total_manifold_dim": int(total_mdim),
    }


# --------------------------------------------------------------------------- #
# Self-test on synthetic data with a KNOWN superposition level.
# --------------------------------------------------------------------------- #
def _selftest():
    """Two regimes validate that excess-interference detects genuine superposition.

    ORTHO  : independent attrs on orthogonal directions -> interference ~ baseline,
             excess ~ 0  (no structured overlap; the probe should NOT cry wolf).
    SUPERP : correlated attrs sharing a packed subspace -> interference >> baseline,
             excess > 0  (the probe detects the superposition).
    Both should recover the concept with high held-out R^2.
    """
    rng = np.random.default_rng(0)
    n = 1500
    for regime in ("ORTHO", "SUPERP"):
        print(f"\n=== regime {regime} ===")
        print(f"{'k':>4} {'R2_a':>6} {'R2_b':>6} {'inter(a,b)':>10} {'baseline':>9} "
              f"{'excess':>8} {'verdict':>8}")
        for k in [64, 16, 8, 4]:
            a = rng.normal(size=n)
            if regime == "ORTHO":
                b = rng.normal(size=n)                       # independent
                Ua = np.zeros(k); Ua[0] = 1.0                # orthogonal axes
                Ub = np.zeros(k); Ub[min(1, k - 1)] = 1.0
            else:
                b = 0.7 * a + np.sqrt(1 - 0.49) * rng.normal(size=n)  # corr 0.7
                v = rng.normal(size=k); v /= np.linalg.norm(v)        # one shared dir
                Ua = v; Ub = v                                        # packed together
            X = np.outer(a, Ua) + np.outer(b, Ub) + 0.05 * rng.normal(size=(n, k))
            rep = superposition_report(X, {"a": a, "b": b}, dim=k, degree=3)
            iab = rep["interference"][0, 1]
            exc = rep["excess_interference"]
            r2a = rep["probes"]["a"]["r2_max"]; r2b = rep["probes"]["b"]["r2_max"]
            verdict = "super" if exc > 0.15 else "clean"
            print(f"{k:>4} {r2a:>6.2f} {r2b:>6.2f} {iab:>10.3f} "
                  f"{rep['random_baseline_cos2']:>9.3f} {exc:>8.3f} {verdict:>8}")
    print("\nexpect: ORTHO excess~0 (clean) at all k; SUPERP excess>>0 (super).")


if __name__ == "__main__":
    _selftest()
