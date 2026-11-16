"""Horizon checks for theory.md, section 9 (H > 1).

Tree MDP: binary actions, state = action prefix, public deterministic dynamics. Teacher log-probs are
released at every (h, s, a) =: z with precision mu^2 * w(z), i.e. y(z) = log pi*(z) + N(0, sigma^2 / w(z)),
sigma = Delta / mu, w a probability over z. Everything below is exact (Gaussian) unless marked MC.

Two selectors:
  plug-in ERM : argmin_pi  sum_z nu^pi(z) (log pi(z) - y(z))          (noisy reverse KL)
  log-space LS: argmin_pi  sum_z w(z) (y(z) - log pi(z))^2            (MLE for the release channel)

Checks
  A. Inequality used in Thm 10: KL(p||q) <= sum_a max(p_a, q_a) log(p_a/q_a)^2.
  B. Two-policy product-tilt instance: worst-case expected risk over the tilt size, vs H.
     Theory: plug-in ERM ~ H^2 sigma^2 (any design), LS ~ H sigma^2.
  C. MC confirmation of the two error-probability formulas at one (H, rho).
  D. 64-policy class: mean risk of both selectors vs H.
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(20261113)


def make_policy(base, rho, signs):
    """Per level h (0-based, 2^h prefixes): P(a = 1 | prefix) = sigmoid(base + 2 rho sign)."""
    return [1.0 / (1.0 + np.exp(-(b + 2.0 * rho * s))) for b, s in zip(base, signs)]


def flatten(p1):
    """Return (d, nu, logp): state occupancy per level, and flat occupancy / log-prob over z = (h, s, a)."""
    d = [np.ones(1)]
    for h in range(len(p1) - 1):
        nxt = np.empty(2 * d[h].size)
        nxt[0::2] = d[h] * (1.0 - p1[h])
        nxt[1::2] = d[h] * p1[h]
        d.append(nxt)
    act = [np.stack([1.0 - p, p], axis=1) for p in p1]
    nu = np.concatenate([(dh[:, None] * a).ravel() for dh, a in zip(d, act)])
    logp = np.concatenate([np.log(a).ravel() for a in act])
    return d, act, nu, logp


def design(policies_flat, H):
    """w(z) = d_mix(s) * qbar(a|s) / H: trajectory-mixture state occupancy times averaged action law."""
    n = len(policies_flat)
    parts = []
    for h in range(H):
        d_mix = sum(pf[0][h] for pf in policies_flat) / n
        q_bar = sum(pf[1][h] for pf in policies_flat) / n
        parts.append((d_mix[:, None] * q_bar).ravel())
    return np.concatenate(parts) / H


# ---------------------------------------------------------------- A
def check_jeffreys(d=30, trials=200_000):
    worst = 0.0
    for _ in range(trials // 1000):
        P = np.maximum(rng.dirichlet(np.full(d, rng.choice([0.05, 0.3, 1.0, 5.0])), size=1000), 1e-300)
        Q = np.maximum(rng.dirichlet(np.full(d, rng.choice([0.05, 0.3, 1.0, 5.0])), size=1000), 1e-300)
        u = np.log(P) - np.log(Q)
        worst = max(worst, float((np.sum(P * u, axis=1) / np.sum(np.maximum(P, Q) * u**2, axis=1)).max()))
    print(f"A. max of KL(p||q) / sum max(p,q) log(p/q)^2 over {trials} random pairs: {worst:.4f}  (must be <= 1)")


# ---------------------------------------------------------------- B, C
def two_policy(H, rho, sigma, base, signs):
    t = flatten(make_policy(base, 0.0, signs))
    c = flatten(make_policy(base, rho, signs))
    w = design([t, c], H)
    gap = float(np.sum(c[2] * (c[3] - t[3])))                    # L(pi') = KL(P^pi' || P^teacher)
    s_erm = sigma * np.sqrt(np.sum((c[2] - t[2]) ** 2 / w))      # sd of plug-in difference, design w
    s_opt = sigma * np.sum(np.abs(c[2] - t[2]))                  # same, best possible design (Lemma 2)
    D = np.sqrt(np.sum(w * (c[3] - t[3]) ** 2))                  # log-space distance D_w
    kappa = float(np.max(np.concatenate(
        [(dh[:, None] * np.maximum(a, ta)).ravel() for dh, a, ta in zip(c[0], c[1], t[1])]) / (H * w)))
    return dict(gap=gap, p_erm=norm.cdf(-gap / s_erm), p_opt=norm.cdf(-gap / s_opt),
                p_ls=norm.cdf(-D / (2 * sigma)), kappa=kappa, t=t, c=c, w=w)


def check_two_policy(sigma=0.01, Hs=(2, 4, 6, 8, 10, 12, 14)):
    print("B. two-policy product tilt, sigma = Delta/mu = 0.01; sup over tilt of E[risk] / sigma^2")
    print("     H    plug-in ERM   plug-in, best design   log-space LS   coverage kappa")
    rows = []
    for H in Hs:
        base = [0.5 * rng.normal(size=2**h) for h in range(H)]
        signs = [rng.choice([-1.0, 1.0], size=2**h) for h in range(H)]
        best = np.zeros(3); kap = 0.0
        for rho in np.geomspace(1e-3, 0.5, 120):
            r = two_policy(H, rho, sigma, base, signs)
            best = np.maximum(best, r["gap"] * np.array([r["p_erm"], r["p_opt"], r["p_ls"]]))
            kap = max(kap, r["kappa"])
        rows.append(best / sigma**2)
        print(f"   {H:3d}   {rows[-1][0]:11.2f}   {rows[-1][1]:20.2f}   {rows[-1][2]:12.2f}   {kap:10.2f}")
    rows = np.array(rows)
    lh = np.log(np.array(Hs, dtype=float))
    sl = [np.polyfit(lh[2:], np.log(rows[2:, j]), 1)[0] for j in range(3)]
    print(f"   log-log slope in H (H >= 6): plug-in {sl[0]:.2f}, plug-in best design {sl[1]:.2f}, LS {sl[2]:.2f}"
          "   (theory 2, 2, 1)")

    # C. Monte Carlo confirmation of the error-probability formulas
    H, rho = 6, 0.03
    base = [0.5 * rng.normal(size=2**h) for h in range(H)]
    signs = [rng.choice([-1.0, 1.0], size=2**h) for h in range(H)]
    r = two_policy(H, rho, sigma, base, signs)
    t, c, w = r["t"], r["c"], r["w"]
    y = t[3] + rng.normal(size=(200_000, w.size)) * sigma / np.sqrt(w)
    erm_wrong = (y @ (c[2] - t[2]) > np.sum(c[2] * c[3]) - np.sum(t[2] * t[3])).mean()
    ls_wrong = (((y - c[3]) ** 2) @ w < ((y - t[3]) ** 2) @ w).mean()
    print(f"C. MC (H=6, rho=0.03): plug-in error {erm_wrong:.4f} vs formula {r['p_erm']:.4f};"
          f" LS error {ls_wrong:.4f} vs formula {r['p_ls']:.4f}")


# ---------------------------------------------------------------- D
def check_class(sigma=0.01, n_pol=64, trials=1500, Hs=(2, 4, 6, 8, 10, 12)):
    print(f"D. class of {n_pol} policies (teacher + tilts, tilt sizes geomspace(0.003, 0.3)), sigma = 0.01;"
          " mean risk / sigma^2")
    print("     H    plug-in ERM   log-space LS   ratio")
    out = []
    for H in Hs:
        base = [0.5 * rng.normal(size=2**h) for h in range(H)]
        flats = [flatten(make_policy(base, 0.0, [np.zeros(2**h) for h in range(H)]))]
        for rho in np.geomspace(0.003, 0.3, n_pol - 1):
            flats.append(flatten(make_policy(base, rho, [rng.choice([-1.0, 1.0], size=2**h) for h in range(H)])))
        NU = np.array([f[2] for f in flats]); LP = np.array([f[3] for f in flats])
        w = design(flats, H)
        risk = np.sum(NU * (LP - LP[0]), axis=1)
        y = LP[0] + rng.normal(size=(trials, w.size)) * sigma / np.sqrt(w)
        erm = np.sum(NU * LP, axis=1)[None, :] - y @ NU.T
        ls = np.sum(w * LP**2, axis=1)[None, :] - 2.0 * y @ (LP * w).T
        r_erm = risk[np.argmin(erm, axis=1)].mean() / sigma**2
        r_ls = risk[np.argmin(ls, axis=1)].mean() / sigma**2
        out.append((r_erm, r_ls))
        print(f"   {H:3d}   {r_erm:11.2f}   {r_ls:12.2f}   {r_erm / r_ls:5.2f}")
    out = np.array(out); lh = np.log(np.array(Hs, dtype=float))
    print(f"   log-log slope in H (H >= 4): plug-in {np.polyfit(lh[1:], np.log(out[1:, 0]), 1)[0]:.2f},"
          f" LS {np.polyfit(lh[1:], np.log(out[1:, 1]), 1)[0]:.2f}")


if __name__ == "__main__":
    check_jeffreys()
    check_two_policy()
    check_class()
