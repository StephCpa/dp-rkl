"""Checks for theory.md, section 13 (answer to the second external review).

  A. Class complexity. An H = 1 class with pathwise KL < 1 whose optimal design quantity is about d / log d,
     so a bounded pathwise KL does not control Gamma_T. Also the envelope bound
     min_w Gamma_T(w) <= E(Pi) := sum_z max_pi nu^pi(z), and a dual certificate for the optimal design.
  C. Anchor-normalized plug-in (Proposition 16): exact check of its variance bound, and the anchor term A.
  B. Paired experiment, design x selector x candidate class, with common random numbers and several
     independent class seeds. The class seed is the statistical unit; intervals are paired t-intervals.
     Designs:   mixprod  = d_mix (x) q_bar / H          (used in earlier sections)
                occmix   = true mixture of the candidates' occupancies
                envelope = proportional to max_pi nu^pi
                optimal  = minimax design for Gamma_T (90%) mixed with occmix (10%) to keep full support
     Selectors: plug-in, plug-in with anchor-linearized normalization, log-space least squares, tournament.

Everything public (candidates, designs) is computed without the teacher. The classes themselves are still
generated around the teacher, so this is a conditional channel experiment, not an end-to-end DP pipeline.
"""
import numpy as np
import check_tournament as ct
import check_hybrid as ch
from check_tournament import Policy, pathwise_max
from _checks import Checker

chk = Checker("check_round2")
T975 = {4: 2.776, 7: 2.365, 9: 2.262, 11: 2.201}       # two-sided 95% t quantiles by degrees of freedom


# ---------------------------------------------------------------- public pair structure and designs
def pair_struct(pols):
    """Public quantities only. Degenerate pairs (identical on their shared mass) are kept for the
    tournament, where their statistic is exactly zero, and excluded from Gamma_T (their S is zero)."""
    n = len(pols)
    pairs, C, const, S = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            m = np.minimum(pols[i].dS, pols[j].dS)
            diff = pols[i].A - pols[j].A
            pairs.append((i, j))
            C.append((m[:, None] * diff).ravel())
            const.append(m @ (pols[i].ent - pols[j].ent))
            S.append(m @ np.sum(diff**2 / np.maximum(pols[i].A, pols[j].A), axis=1))
    return np.array(pairs), np.array(C), np.array(const), np.array(S)


def gamma_T(C, S, w):
    ok = S > 1e-14
    return float(((C[ok] ** 2 @ (1.0 / w)) / S[ok]).max())


def optimal_design(C, S, iters=3000):
    """min_w max_p sum_z C_p(z)^2 / (w(z) S_p), with a dual certificate.
    For pair weights lam on the simplex, min_w sum_z b(z)/w(z) = (sum_z sqrt b(z))^2 with b = lam @ A, so
    (sum_z sqrt(lam @ A))^2 is a lower bound on the minimax value for every lam. Exponentiated-gradient
    ascent on that concave dual; the primal iterate is the best response w ∝ sqrt(lam @ A)."""
    ok = S > 1e-14
    A = C[ok] ** 2 / S[ok][:, None]
    lam = np.full(A.shape[0], 1.0 / A.shape[0])
    ub, lb, best_w = np.inf, 0.0, None
    lam_avg = np.zeros_like(lam)
    for it in range(iters):
        b = lam @ A
        rb = np.sqrt(b)
        lb = max(lb, float(rb.sum() ** 2))
        w = (rb + 1e-300) / (rb.sum() + 1e-300 * rb.size)
        vals = A @ (1.0 / w)
        if vals.max() < ub:
            ub, best_w = float(vals.max()), w
        lam_avg += lam
        if it % 50 == 49:                                   # averaged dual iterate: often a better certificate
            la = lam_avg / (it + 1)
            ra = np.sqrt(la @ A)
            lb = max(lb, float(ra.sum() ** 2))
            wa = (ra + 1e-300) / (ra.sum() + 1e-300 * ra.size)
            va = float((A @ (1.0 / wa)).max())
            if va < ub:
                ub, best_w = va, wa
        eta = 2.0 / np.sqrt(it + 1.0)
        lam = lam * np.exp(eta * (vals / vals.max() - 1.0))
        lam /= lam.sum()
    return best_w, ub, lb


def designs_for(pols, C, S):
    H = pols[0].H
    NU = np.array([p.nu for p in pols])
    occmix = NU.mean(axis=0) / H
    env = NU.max(axis=0)
    w_opt, ub, lb = optimal_design(C, S)
    out = {
        "mixprod": ct.design(pols),
        "occmix": occmix,
        "envelope": env / env.sum(),
        "optimal": 0.9 * w_opt + 0.1 * occmix,
    }
    return out, float(env.sum()), ub, lb


# ---------------------------------------------------------------- A
def check_complexity():
    print("A. H = 1 class with pathwise KL < 1 and a large optimal design quantity")
    print("      d    max KL to teacher   min_w Gamma_T: upper   dual lower   gap     d/log d   envelope mass E")
    for d in (16, 64, 256):
        q = 1.0 / np.log(d)
        teacher = Policy([np.full((1, d), 1.0 / d)])
        pols = []
        for j in range(d):
            for mass in (q, 1.01 * q):
                a = np.full(d, (1.0 - mass) / (d - 1)); a[j] = mass
                pols.append(Policy([a[None, :]]))
        if d > 64:                                          # all pairs would be large; keep within-j and a sample
            keep = [2 * j for j in range(d)] + [2 * j + 1 for j in range(0, d, 4)]
            pols = [pols[i] for i in sorted(set(keep))]
        B = max(pathwise_max(p, teacher) for p in pols)
        pairs, C, const, S = pair_struct(pols)
        w, ub, lb = optimal_design(C, S, iters=1500)
        env = float(np.array([p.nu for p in pols]).max(axis=0).sum())
        print(f"   {d:4d}   {B:14.3f}   {ub:18.2f}   {lb:10.2f}   {100 * (ub - lb) / ub:5.2f}%   {d / np.log(d):7.2f}"
              f"   {env:10.2f}")
        chk.le(f"A d={d}: pathwise KL < 1", B, 1.0)
        chk.le(f"A d={d}: dual lower bound <= primal upper bound", lb, ub, tol=1e-6 * ub)
        w_env = np.array([p.nu for p in pols]).max(axis=0); w_env = w_env / w_env.sum()
        chk.le(f"A d={d}: certified lower bound <= envelope mass", lb, env, tol=1e-9 * env)
        chk.le(f"A d={d}: Gamma_T(envelope design) <= envelope mass", gamma_T(C, S, w_env), env, tol=1e-9 * env)
        chk.between(f"A d={d}: optimal Gamma_T / (d/log d)", lb / (d / np.log(d)), 0.5, 2.0)


# ---------------------------------------------------------------- B
def selectors(pols, t, sigma, g, w, pairs, C, const):
    """Risks of the four selectors for one design, given standard-normal draws g shared across designs."""
    n, H = len(pols), pols[0].H
    k = pols[0].A.shape[1]
    NU = np.array([p.nu for p in pols]); LP = np.array([p.logp for p in pols])
    risk = np.sum(NU * (LP - t.logp), axis=1)
    y = t.logp + sigma * g / np.sqrt(w)
    c_erm = np.sum(NU * LP, axis=1)
    i_erm = np.argmin(c_erm[None, :] - y @ NU.T, axis=1)
    # anchor-linearized normalization: remove, state by state, the w-weighted projection of the release
    # onto the anchor q_bar, treating <q_bar, log p* - log q_bar> as zero (first-order normalization)
    qbar = np.mean([p.A for p in pols], axis=0)                               # (n_states, k), public
    ws = w.reshape(-1, k)
    denom = np.sum(qbar**2 / ws, axis=1)
    alpha = np.einsum("nsa,sa->ns", NU.reshape(n, -1, k), qbar / ws) / denom  # (n, n_states)
    NUn = (NU.reshape(n, -1, k) - alpha[:, :, None] * qbar[None]).reshape(n, -1)
    c_n = c_erm - alpha @ np.sum(qbar * np.log(qbar), axis=1)
    i_norm = np.argmin(c_n[None, :] - y @ NUn.T, axis=1)
    i_ls = np.argmin(np.sum(w * LP**2, axis=1)[None, :] - 2.0 * y @ (LP * w).T, axis=1)
    D = const[None, :] - y @ C.T
    full = np.zeros((g.shape[0], n, n))
    full[:, pairs[:, 0], pairs[:, 1]] = D
    full[:, pairs[:, 1], pairs[:, 0]] = -D
    full[:, np.arange(n), np.arange(n)] = -np.inf
    i_tr = np.argmin(full.max(axis=2), axis=1)
    out = {"plug-in": risk[i_erm], "plug-in+norm": risk[i_norm], "least squares": risk[i_ls],
           "tournament": risk[i_tr]}
    for name, r in out.items():
        chk.finite(f"B risk {name}", r)
    return out, risk


def make_class(kind, seed):
    ct.rng = np.random.default_rng(seed)
    ch.rng = np.random.default_rng(seed + 10_000)
    if kind[0] == "realizable":
        _, H, n_pol, rho_max = kind
        base = [0.5 * ct.rng.normal(size=2**h) for h in range(H)]
        t = ct.binary_policy(base, 0.0)
        return t, [t] + [ct.binary_policy(base, r) for r in np.geomspace(0.003, rho_max, n_pol - 1)]
    if kind[0] == "misspecified":
        return ch.misspecified_class(kind[1])
    return ch.trap_class(kind[1])


def paired(a, b):
    """Mean of a - b over seeds with a 95% paired t-interval."""
    d = np.asarray(a) - np.asarray(b)
    half = T975[len(d) - 1] * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - half, d.mean() + half


def check_paired(n_seeds=10, trials=300):
    configs = [
        (("realizable", 8, 16, 0.3), 0.01, "sigma^2"),
        (("realizable", 8, 64, 0.3), 0.01, "sigma^2"),
        (("realizable", 8, 64, 1.0), 0.01, "sigma^2"),
        (("misspecified", 6), 0.05, "L*"),
        (("trap", 8), 0.03, "L*"),
    ]
    design_names = ["mixprod", "occmix", "envelope", "optimal"]
    sel_names = ["plug-in", "plug-in+norm", "least squares", "tournament"]
    print(f"\nB. paired experiment: {n_seeds} class seeds x {trials} noise draws (shared across designs)")
    for kind, sigma, unit in configs:
        res = {dn: {sn: [] for sn in sel_names} for dn in design_names}
        gam = {dn: [] for dn in design_names}
        envs, gaps, Bs = [], [], []
        for seed in range(n_seeds):
            t, pols = make_class(kind, 1000 + seed)
            pairs, C, const, S = pair_struct(pols)
            dsg, env, ub, lb = designs_for(pols, C, S)
            envs.append(env / pols[0].H); gaps.append((ub - lb) / ub)
            Bs.append(max(pathwise_max(p, t) for p in pols))
            chk.le(f"B {kind} seed {seed}: dual <= primal", lb, ub, tol=1e-6 * ub)
            g = np.random.default_rng(50_000 + seed).normal(size=(trials, pols[0].nu.size))
            for dn in design_names:
                w = dsg[dn]
                chk.finite(f"B design {dn}", w)
                chk.close(f"B design {dn} sums to 1", w.sum(), 1.0, rel=1e-9)
                gam[dn].append(gamma_T(C, S, w))
                out, risk = selectors(pols, t, sigma, g, w, pairs, C, const)
                scale = sigma**2 if unit == "sigma^2" else risk.min()
                for sn in sel_names:
                    res[dn][sn].append(out[sn].mean() / scale)
        H = kind[1]
        print(f"\n   class {kind}, sigma = {sigma}; risk / {unit}, mean over seeds (se);"
              f" envelope mass / H = {np.mean(envs):.2f}, max pathwise KL = {np.mean(Bs):.2f},"
              f" dual gap of the optimal design <= {100 * max(gaps):.2f}%")
        print("      design     Gamma_T/H   " + "   ".join(f"{sn:>15s}" for sn in sel_names))
        for dn in design_names:
            print(f"      {dn:9s}  {np.mean(gam[dn]) / H:8.2f}   " + "   ".join(
                f"{np.mean(res[dn][sn]):8.2f} ({np.std(res[dn][sn], ddof=1) / np.sqrt(n_seeds):5.2f})"
                for sn in sel_names))
        print("      paired differences (mean [95% interval], negative = first is better):")
        for dn in design_names[1:]:
            m, lo, hi = paired(res[dn]["tournament"], res["mixprod"]["tournament"])
            print(f"         tournament, {dn:8s} - mixprod: {m:8.2f} [{lo:8.2f}, {hi:8.2f}]")
        for a, b in (("tournament", "least squares"), ("tournament", "plug-in"), ("plug-in+norm", "plug-in"),
                     ("plug-in+norm", "tournament")):
            m, lo, hi = paired(res["mixprod"][a], res["mixprod"][b])
            print(f"         mixprod, {a:13s} - {b:13s}: {m:8.2f} [{lo:8.2f}, {hi:8.2f}]")


# ---------------------------------------------------------------- C
def check_anchor(n_seeds=4):
    """Product-form design w(s,a) = omega(s) q_bar(a|s): the normalized plug-in has coefficient
    d^pi(s) (pi - q_bar)(a|s) and estimates J(pi) = L(pi) - C(pi), C(pi) = sum_s d^pi(s) KL(q_bar_s || pi*_s).
    Check Var[J_hat(pi)] <= 8 kappa H sigma^2 (L(pi) + C(pi)) with the public kappa
    = max_z d^pi(s) max(pi, q_bar)(a|s) / (H w(z)), and report the anchor term A = max_pi C(pi)."""
    print("\nC. anchor-normalized plug-in: variance bound (max ratio must be <= 1) and anchor term A")
    print("      class                         max Var / bound    kappa    A / L*   (L* = 0 when realizable)")
    for kind in (("realizable", 8, 64, 0.3), ("realizable", 8, 64, 1.0), ("misspecified", 6), ("trap", 8)):
        worst, kaps, ratios = 0.0, [], []
        for seed in range(n_seeds):
            t, pols = make_class(kind, 1000 + seed)
            H, k = pols[0].H, pols[0].A.shape[1]
            w = ct.design(pols).reshape(-1, k)
            qbar = np.mean([p.A for p in pols], axis=0)
            kl_anchor = np.sum(qbar * (np.log(qbar) - np.log(t.A)), axis=1)
            risks, Cs = [], []
            kap = max(float(((p.dS[:, None] * np.maximum(p.A, qbar)) / (H * w)).max()) for p in pols)
            for p in pols:
                L = float(np.sum(p.nu * (p.logp - t.logp)))
                Cp = float(p.dS @ kl_anchor)
                var = float(np.sum((p.dS[:, None] * (p.A - qbar)) ** 2 / w))          # per unit sigma^2
                worst = max(worst, var / (8 * kap * H * (L + Cp) + 1e-300))
                risks.append(L); Cs.append(Cp)
            kaps.append(kap)
            ratios.append(max(Cs) / min(risks) if min(risks) > 1e-12 else np.nan)
        chk.le(f"C {kind}: variance bound", worst, 1.0)
        tail = "   n/a" if np.all(np.isnan(ratios)) else f"{np.nanmean(ratios):6.2f}"
        print(f"      {str(kind):28s}  {worst:12.4f}   {np.mean(kaps):6.2f}   {tail}")


if __name__ == "__main__":
    check_complexity()
    check_anchor()
    check_paired()
    chk.finish()
