"""Public pair structure and the tournament design quantity Gamma_T, shared by the check scripts.

For a pair p = (pi, pi') with shared occupancy m(s) = min(d^pi(s), d^pi'(s)):
    c_p(z) = m(s) (pi - pi')(a|s),      S_p = sum_s m(s) sum_a (pi - pi')^2 / max(pi, pi'),
    Gamma_T(w) = max_p [ sum_z c_p(z)^2 / w(z) ] / S_p.

The ratio is invariant to rescaling pi - pi', so a nearly duplicate pair has a perfectly well-defined ratio
and must be kept: dropping pairs with small S (as an earlier version did with S > 1e-14) can hide the pair
that determines Gamma_T. Each pair is therefore rescaled by max |pi - pi'| before the ratio is formed. Only
pairs that are exactly identical on their shared mass are excluded: their statistic is identically zero with
zero variance, and the ratio is 0/0. If no pair remains, Gamma_T is defined as 0.
"""
import numpy as np


def pair_struct(pols):
    """Returns pairs, raw C and const (for the tournament statistic), and scale-normalized Cn, Sn with a
    validity mask (for Gamma_T and the design problem). Uses only the candidates: everything is public."""
    n = len(pols)
    pairs, C, const, Cn, Sn = [], [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            m = np.minimum(pols[i].dS, pols[j].dS)
            diff = pols[i].A - pols[j].A
            pairs.append((i, j))
            C.append((m[:, None] * diff).ravel())
            const.append(m @ (pols[i].ent - pols[j].ent))
            scale = float(np.abs(diff[m > 0]).max()) if np.any(m > 0) else 0.0
            dn = diff / scale if scale > 0 else np.zeros_like(diff)
            Cn.append((m[:, None] * dn).ravel())
            Sn.append(m @ np.sum(dn**2 / np.maximum(pols[i].A, pols[j].A), axis=1))
    width = pols[0].nu.size
    as2d = lambda rows: np.array(rows).reshape(len(rows), width)
    Sn = np.array(Sn, dtype=float)
    return (np.array(pairs, dtype=int).reshape(-1, 2), as2d(C), np.array(const, dtype=float),
            as2d(Cn), Sn, Sn > 0)


def gamma_T(Cn, Sn, valid, w):
    if not np.any(valid):
        return 0.0
    return float(((Cn[valid] ** 2 @ (1.0 / w)) / Sn[valid]).max())


def optimal_design(Cn, Sn, valid, fallback, iters=3000):
    """min_w max_p sum_z c_p(z)^2 / (w(z) S_p) over valid pairs, with a dual certificate.
    For pair weights lam on the simplex, min_w sum_z b(z)/w(z) = (sum_z sqrt b(z))^2 with b = lam @ A, so
    (sum_z sqrt(lam @ A))^2 lower-bounds the minimax value for every lam. Exponentiated-gradient ascent on
    that concave dual; the primal iterate is the best response w ∝ sqrt(lam @ A).
    Returns (w, upper bound = Gamma_T(w), certified lower bound). With no valid pair: (fallback, 0, 0)."""
    if not np.any(valid):
        return fallback, 0.0, 0.0
    A = Cn[valid] ** 2 / Sn[valid][:, None]
    lam = np.full(A.shape[0], 1.0 / A.shape[0])
    ub, lb, best_w = np.inf, 0.0, fallback
    lam_avg = np.zeros_like(lam)
    for it in range(iters):
        rb = np.sqrt(lam @ A)
        lb = max(lb, float(rb.sum() ** 2))
        w = (rb + 1e-300) / (rb.sum() + 1e-300 * rb.size)
        vals = A @ (1.0 / w)
        if vals.max() < ub:
            ub, best_w = float(vals.max()), w
        lam_avg += lam
        if it % 50 == 49:
            ra = np.sqrt((lam_avg / (it + 1)) @ A)
            lb = max(lb, float(ra.sum() ** 2))
            wa = (ra + 1e-300) / (ra.sum() + 1e-300 * ra.size)
            va = float((A @ (1.0 / wa)).max())
            if va < ub:
                ub, best_w = va, wa
        lam = lam * np.exp(2.0 / np.sqrt(it + 1.0) * (vals / vals.max() - 1.0))
        lam /= lam.sum()
    return best_w, ub, lb
