"""Numerical checks for theory.md.

Model: teacher log-probs ell in R^d with coordinatewise (l_inf) sensitivity Delta.
A release with precision shares w (sum_a w_a = beta) outputs ell_a + N(0, Delta^2 / w_a)
and costs sqrt(beta)-GDP. A sampled release draws a ~ r and outputs ell_a + N(0, Delta^2/beta).

Checks
  1. Design identity (Lemma 1): V_samp(c; r) = V_det(c; beta r) + sampling variance.
  2. Optimal design for one functional (Lemma 2): min_w V_det = Delta^2 ||c||_1^2 / beta.
  3. Gamma(C) bounds (Prop 3): 1 <= Gamma <= min{d, |C|, e^b}; Gamma = d for deterministic class.
  4. Inequality used in Thm 5: sum (pi - pi*)^2 / max(pi, pi*) <= 4 KL(pi || pi*).
  5. Gradient-noise trace (Prop 7): optimal design vs sampled token vs homoscedastic.
  6. Selection simulation (Thm 5 / Prop 6'): privacy floor is T-independent, scales as (Delta/mu)^2,
     and on this benign instance sampled feedback reaches the weighted-full floor as T grows.
     (Not true in general: see the rare-action instance in check_review.py.)
"""
import numpy as np

rng = np.random.default_rng(20261112)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


# ---------------------------------------------------------------- 1
def check_design_identity(d=12, beta=0.7, Delta=0.3, n=600_000):
    ell = np.log(softmax(1.5 * rng.normal(size=d)))
    c = softmax(rng.normal(size=d))
    r = softmax(0.5 * rng.normal(size=d))
    sig2 = Delta**2 / (beta * r)
    est_det = (ell + rng.normal(size=(n, d)) * np.sqrt(sig2)) @ c
    a = rng.choice(d, size=n, p=r)
    est_s = c[a] / r[a] * (ell[a] + rng.normal(size=n) * Delta / np.sqrt(beta))
    v_det = Delta**2 / beta * np.sum(c**2 / r)
    v_sampling = np.sum(r * (c * ell / r) ** 2) - (c @ ell) ** 2
    print("1. design identity")
    print(f"   det : MC var {est_det.var():.5f}  theory {v_det:.5f}  bias {est_det.mean() - c @ ell:+.4f}")
    print(f"   samp: MC var {est_s.var():.5f}  theory {v_det + v_sampling:.5f}"
          f"  (privacy {v_det:.5f} + sampling {v_sampling:.5f})")


# ---------------------------------------------------------------- 2
def check_optimal_design(d=40, trials=20000):
    c = softmax(2.0 * rng.normal(size=d))
    opt = np.sum(np.abs(c)) ** 2            # at w ∝ |c|
    uni = d * np.sum(c**2)
    best_random = min(np.sum(c**2 / w) for w in rng.dirichlet(np.ones(d), size=trials))
    w_star = np.abs(c) / np.abs(c).sum()
    print("2. optimal design for one functional (units of Delta^2/beta)")
    print(f"   w∝|c|: {np.sum(c**2 / w_star):.4f} (= ||c||_1^2 = {opt:.4f});"
          f" uniform: {uni:.3f}; best of {trials} random designs: {best_random:.3f}")


# ---------------------------------------------------------------- 3
def gamma_numeric(C):
    """min_{w in simplex} max_{c in C} sum_a c_a^2 / w_a  (convex in w).
    L-BFGS on an annealed log-sum-exp surrogate with w = softmax(z), warm-started from the best of
    three simple designs. Returns the best exact max found, i.e. an upper bound on Gamma."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    n, d = C.shape
    C2 = C**2
    exact = lambda w: float((C2 @ (1.0 / w)).max())
    starts = [np.full(d, 1.0 / d), C.mean(axis=0), np.sqrt(C2.max(axis=0)) / np.sqrt(C2.max(axis=0)).sum()]
    w = min(starts, key=exact)
    best, z = exact(w), np.log(w)
    for tau in [0.3, 0.1, 0.03, 0.01, 0.003]:
        def f(z, tau=tau):
            w = softmax(z)
            vals = C2 @ (1.0 / w)
            scale = tau * vals.max()
            return scale * logsumexp(vals / scale)
        res = minimize(f, z, method="L-BFGS-B")
        z = res.x
        best = min(best, exact(softmax(z)))
    return best


def check_gamma(d=24):
    print("3. Gamma(C) bounds")
    det = np.eye(d)
    print(f"   deterministic class: Gamma ≈ {gamma_numeric(det):.2f}  (theory d = {d})")
    for n, scale in [(5, 3.0), (200, 3.0), (200, 0.3)]:
        q0 = softmax(rng.normal(size=d))
        C = np.array([softmax(np.log(q0) + scale * rng.normal(size=d)) for _ in range(n)])
        eb = (C / q0).max()
        mix = C.mean(axis=0)
        g = gamma_numeric(C)
        print(f"   n={n:3d} spread={scale}: Gamma ≈ {g:6.2f}; bounds: lower 1, d={d}, |C|={n},"
              f" e^b={eb:8.2f}; design values: uniform {(C**2 @ np.full(d, d)).max():6.2f},"
              f" mixture {(C**2 @ (1 / mix)).max():6.2f}, q0 {(C**2 @ (1 / q0)).max():6.2f}")


# ---------------------------------------------------------------- 4
def check_chi_kl(d=30, trials=200_000):
    worst = 0.0
    for _ in range(trials // 1000):
        P = rng.dirichlet(np.full(d, rng.choice([0.05, 0.3, 1.0, 5.0])), size=1000)
        Q = rng.dirichlet(np.full(d, rng.choice([0.05, 0.3, 1.0, 5.0])), size=1000)
        P = np.maximum(P, 1e-300); Q = np.maximum(Q, 1e-300)
        lhs = np.sum((P - Q) ** 2 / np.maximum(P, Q), axis=1)
        rhs = 4 * np.sum(P * (np.log(P) - np.log(Q)), axis=1)
        worst = max(worst, float((lhs / rhs).max()))
    print(f"4. max over {trials} random pairs of  sum (p-q)^2/max(p,q) / (4 KL(p||q)) = {worst:.4f}  (must be <= 1)")


# ---------------------------------------------------------------- 5
def check_gradient_noise(d=1000):
    print("5. gradient-noise trace per position, units of Delta^2/beta (logit-space score e_a - p)")
    print("   student            sampled   optimal   homosc.full   top16-homosc (tail mass)   sampled/optimal")
    cases = {
        "uniform": np.full(d, 1.0 / d),
        "zipf(1.0)": (lambda z: z / z.sum())(1.0 / np.arange(1, d + 1)),
        "peaked p1=0.9": np.r_[0.9, np.full(d - 1, 0.1 / (d - 1))],
        "peaked p1=0.99": np.r_[0.99, np.full(d - 1, 0.01 / (d - 1))],
    }
    for name, p in cases.items():
        x2 = 1 - 2 * p + np.sum(p**2)          # ||e_a - p||^2
        x = np.sqrt(x2)
        sampled = np.sum(p * x2)               # = 1 - ||p||^2
        optimal = np.sum(p * x) ** 2           # w ∝ p_a ||e_a - p||
        homo = d * np.sum(p**2 * x2)
        top = np.argsort(-p)[:16]
        top16 = 16 * np.sum(p[top] ** 2 * x2[top])
        tail = 1 - p[top].sum()
        print(f"   {name:16s} {sampled:8.4f}  {optimal:8.4f}  {homo:10.4f}   {top16:9.4f} ({tail:.3f})"
              f"            {sampled / optimal:6.2f}")
    # Monte Carlo confirmation for one case
    p = cases["peaked p1=0.9"][:50].copy(); p /= p.sum(); dd = len(p)
    x = np.sqrt(1 - 2 * p + np.sum(p**2)); w = p * x / np.sum(p * x)
    n = 200_000
    xi = rng.normal(size=(n, dd)) / np.sqrt(w)              # Delta^2/beta = 1
    g = (xi * p) @ (np.eye(dd) - p)                         # sum_a p_a xi_a (e_a - p)
    a = rng.choice(dd, size=5_000_000, p=p)
    tr_s = float(np.mean(rng.normal(size=a.size) ** 2 * (x**2)[a]))
    print(f"   MC (d=50, peaked): optimal {np.sum(g.var(axis=0)):.5f} vs theory {np.sum(p * x) ** 2:.5f};"
          f" sampled {tr_s:.5f} vs theory {1 - np.sum(p**2):.5f}")


# ---------------------------------------------------------------- 6
def check_selection(d=64, n_pol=128, trials=4000):
    logits = -1.2 * np.log(np.arange(1, d + 1)) + 0.3 * rng.normal(size=d)
    t = softmax(logits)                                    # teacher, in the class (realizable)
    rhos = np.geomspace(0.02, 1.5, n_pol - 1)
    Pi = [t] + [softmax(np.log(t) + rho * rng.choice([-1.0, 1.0], size=d)) for rho in rhos]
    Pi = np.array(Pi)
    risk = np.array([kl(pi, t) for pi in Pi])
    q = Pi.mean(axis=0)                                    # public design: mixture of candidates
    eb = (np.maximum(Pi, t) / q).max()
    ell = np.log(t)
    ent = np.sum(Pi * np.log(Pi), axis=1)
    iota = np.log(n_pol)
    print(f"6. selection, single public context, realizable; d={d}, |Pi|={n_pol}, e^b={eb:.2f}, log|Pi|={iota:.2f}")
    print("   mean risk KL(pi_hat || teacher); 'bound' = 16 e^b (Delta/mu)^2 log|Pi| (delta dropped)")
    print("   Delta/mu  weighted-full  homosc-full   sampled T=1e1    T=1e2    T=1e3    T=1e5   | bound")
    floors = []
    for dm in [0.0, 0.005, 0.02, 0.05, 0.1, 0.2]:
        # weighted full release: aggregate noise is exactly N(0, (Delta/mu)^2 / q_a), any T
        y = ell + rng.normal(size=(trials, d)) * dm / np.sqrt(q)
        r_w = risk[np.argmin(ent[None, :] - y @ Pi.T, axis=1)].mean()
        y = ell + rng.normal(size=(trials, d)) * dm * np.sqrt(d)
        r_h = risk[np.argmin(ent[None, :] - y @ Pi.T, axis=1)].mean()
        row = []
        for T in [10**1, 10**2, 10**3, 10**5]:
            n = rng.multinomial(T, q, size=trials) / T                       # empirical action freq
            zeta = rng.normal(size=(trials, d)) * dm * np.sqrt(n)            # aggregated privacy noise
            centred = ell - np.log(q)
            corr = (n * centred + zeta) @ (Pi / q).T                         # IW estimate of <pi, ell - log q>
            est = ent[None, :] - (Pi @ np.log(q))[None, :] - corr
            row.append(risk[np.argmin(est, axis=1)].mean())
        if dm >= 0.02:
            floors.append(r_w)
        print(f"   {dm:7.3f}  {r_w:12.5f}  {r_h:11.5f}   " + "  ".join(f"{v:8.5f}" for v in row)
              + f"   | {16 * eb * dm**2 * iota:8.3f}")
    slope = np.polyfit(np.log([0.02, 0.05, 0.1, 0.2]), np.log(floors), 1)[0]
    print(f"   log-log slope of weighted-full floor vs Delta/mu: {slope:.2f}  (theory 2 in the realizable case)")


if __name__ == "__main__":
    check_design_identity()
    check_optimal_design()
    check_gamma()
    check_chi_kl()
    check_gradient_noise()
    check_selection()
