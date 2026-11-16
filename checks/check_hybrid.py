"""Checks for theory.md, section 12 (fourth pass).

  S1. Step-by-step verification of the proof of Lemma 12 on adversarially varied tree MDPs: every inequality
      in the chain is evaluated exactly (no Monte Carlo).
  S2. Trap instance: the tournament alone can select a candidate with risk >> L* even with no noise
      (the factor B in its approximation term is real). The plug-in does not have this failure.
  S3. Hybrid selector: eliminate by plug-in confidence intervals (whose noise level is a public number),
      then run the shared-mass tournament among the survivors. Guarantee: the smaller of the two bounds.
  S4. Design: the public quantity Gamma_T(w) = max_pairs Var-coefficient / S_tilde replaces kappa * H in
      Theorem 14. Mixture design vs the minimax-optimal design, as H grows.
"""
import numpy as np
from check_tournament import Policy, binary_policy, design, local_kl, pair_tables, pathwise_max

rng = np.random.default_rng(20261116)
C0 = (1.0 - 2.0 ** -0.25) ** 2                      # phi(1/sqrt 2), phi(w) = (1 - sqrt w)^2


# ---------------------------------------------------------------- S1
def proof_chain(p, q, t):
    """Exact evaluation of each quantity in the proof of Lemma 12 for the ordered pair (pi = p, pi' = q)."""
    k = p.act[0].shape[1]
    H = p.H
    kl_p = [np.sum(p.act[h] * (np.log(p.act[h]) - np.log(t.act[h])), axis=1) for h in range(H)]
    L_p = sum(float(p.d[h] @ kl_p[h]) for h in range(H))
    L_q = float(np.sum(q.nu * (q.logp - t.logp)))
    T = sum(float(np.minimum(p.d[h], q.d[h]) @ kl_p[h]) for h in range(H))
    alive = np.ones(1, dtype=bool)
    pre_tau_kl = 0.0          # E_pi[ sum_{h < tau} k_h ]
    pre_sigma_kl = 0.0        # E_pi[ sum_{j < sigma} k_j ],  sigma = tau ^ H
    e_phi = 0.0               # E_pi[ phi(W_sigma) ]
    e_neglog = 0.0            # E_pi[ -log W_sigma ]
    pE1 = pE2 = pstar_hit = 0.0
    v_low_alive = np.ones(1, dtype=bool)     # under P*: no state so far with V < 1/sqrt2
    for h in range(H):
        R = q.d[h] / p.d[h]; W = t.d[h] / p.d[h]; V = q.d[h] / t.d[h]
        cross = alive & (R < 0.5)
        pE1 += float(p.d[h][cross & (W < 2 ** -0.5)].sum())
        pE2 += float(p.d[h][cross & (W >= 2 ** -0.5)].sum())
        stopped = cross | (alive & ~cross & (h == H - 1))
        e_phi += float((p.d[h][stopped] * (1 - np.sqrt(W[stopped])) ** 2).sum())
        e_neglog += float((p.d[h][stopped] * -np.log(W[stopped])).sum())
        alive = alive & ~cross
        pre_tau_kl += float((p.d[h][alive] * kl_p[h][alive]).sum())
        if h < H - 1:
            pre_sigma_kl += float((p.d[h][alive] * kl_p[h][alive]).sum())
        hit = v_low_alive & (V < 2 ** -0.5)
        pstar_hit += float(t.d[h][hit].sum())
        v_low_alive = v_low_alive & ~hit
        if h < H - 1:
            alive = np.repeat(alive, k); v_low_alive = np.repeat(v_low_alive, k)
    e_phi_V = float(np.sum((np.sqrt(t.d[H - 1]) - np.sqrt(q.d[H - 1])) ** 2))
    B = pathwise_max(p, t)
    tiny = 1e-300
    return {
        "L <= pre_tau + B*P(cross)": L_p / (pre_tau_kl + B * (pE1 + pE2) + tiny),
        "pre_tau <= 2T": pre_tau_kl / (2 * T + tiny),
        "c0*P(E1) <= E phi(W_sigma)": C0 * pE1 / (e_phi + tiny),
        "E phi(W_sigma) <= E[-log W_sigma]": e_phi / (e_neglog + tiny),
        "E[-log W_sigma] = pre_sigma (abs diff)": abs(e_neglog - pre_sigma_kl),
        "pre_sigma <= pre_tau": pre_sigma_kl / (pre_tau_kl + tiny),
        "P(E2) <= sqrt2 * P*(V low)": pE2 / (2 ** 0.5 * pstar_hit + tiny),
        "c0*P*(V low) <= E* phi(V_H)": C0 * pstar_hit / (e_phi_V + tiny),
        "E* phi(V_H) <= L(pi')": e_phi_V / (L_q + tiny),
        "final: L <= (2+79B)T + 56B L'": L_p / ((2 + 79 * B) * T + 56 * B * L_q + tiny),
    }


def random_policy(H, k, scale, floor=0.0):
    act = []
    for h in range(H):
        g = scale * rng.normal(size=(k**h, k)) * rng.choice([0.2, 1.0, 3.0])
        a = np.exp(g - g.max(axis=1, keepdims=True)); a /= a.sum(axis=1, keepdims=True)
        act.append((1 - floor) * a + floor / k)
    return Policy(act)


def check_proof_chain(n_instances=400):
    worst = {}
    for it in range(n_instances):
        # every policy below must be row-normalized; asserted in the loop
        k = int(rng.choice([2, 3])); H = int(rng.choice([3, 5, 7])) if k == 2 else int(rng.choice([3, 5]))
        t = random_policy(H, k, rng.choice([0.3, 1.0, 2.0]), floor=1e-3)
        mode = it % 4
        if mode == 0:      # both near the teacher
            def mk():
                acts = []
                for a in t.act:
                    b = a * np.exp(0.3 * rng.normal(size=a.shape))
                    acts.append(b / b.sum(axis=1, keepdims=True))
                return Policy(acts)
            p, q = mk(), mk()
        elif mode == 1:    # unrelated, possibly near-deterministic
            p, q = random_policy(H, k, 3.0, 1e-4), random_policy(H, k, 3.0, 1e-4)
        elif mode == 2:    # pi' is the teacher with one branch nearly removed (trap shape)
            q_act = [a.copy() for a in t.act]
            q_act[0][0, 0] = 1e-3 * q_act[0][0, 0]; q_act[0] /= q_act[0].sum(axis=1, keepdims=True)
            q = Policy(q_act); p = random_policy(H, k, 2.0, 1e-4)
        else:              # pi equals the teacher except deep in the tree
            p_act = [a.copy() for a in t.act]
            g = rng.normal(size=p_act[-1].shape) * 2.0
            p_act[-1] = p_act[-1] * np.exp(g); p_act[-1] /= p_act[-1].sum(axis=1, keepdims=True)
            p = Policy(p_act); q = random_policy(H, k, 1.0, 1e-4)
        for pol_ in (p, q, t):
            assert all(np.allclose(a.sum(axis=1), 1.0) for a in pol_.act)
        for a, b in ((p, q), (q, p)):
            for name, val in proof_chain(a, b, t).items():
                worst[name] = max(worst.get(name, 0.0), val)
    print(f"S1. proof of Lemma 12, worst value of each step over {2 * n_instances} ordered pairs"
          " (ratios LHS/RHS must be <= 1; the identity must be ~0)")
    for name, val in worst.items():
        print(f"      {name:42s} {val:.3e}" if "abs diff" in name else f"      {name:42s} {val:.4f}")


# ---------------------------------------------------------------- selectors
def run_selectors(pols, t, sigma, trials, w, delta=0.1):
    n = len(pols)
    NU = np.array([p.nu for p in pols]); LP = np.array([p.logp for p in pols])
    risk = np.sum(NU * (LP - t.logp), axis=1)
    T, pairs, C, const, kap = pair_tables(pols, t, w)
    y = t.logp + rng.normal(size=(trials, w.size)) * sigma / np.sqrt(w)
    Lhat = np.sum(NU * LP, axis=1)[None, :] - y @ NU.T
    i_erm = np.argmin(Lhat, axis=1)
    D = const[None, :] - y @ C.T
    full = np.zeros((trials, n, n))
    full[:, pairs[:, 0], pairs[:, 1]] = D
    full[:, pairs[:, 1], pairs[:, 0]] = -D
    diag = np.arange(n)
    plain = full.copy(); plain[:, diag, diag] = -np.inf
    i_tr = np.argmin(plain.max(axis=2), axis=1)
    # hybrid: plug-in elimination with exactly known noise levels, then tournament among survivors
    G = (NU / w) @ NU.T
    s = sigma * np.sqrt(np.maximum(np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G, 0.0))
    z = np.sqrt(2 * np.log(n * n / delta))
    out, surv = [], None
    for zz in (z, 1.0):                                            # proof-calibrated threshold, then z = 1
        elim = ((Lhat[:, :, None] - Lhat[:, None, :] - zz * s[None, :, :]) > 0).any(axis=2)   # (trials, n)
        hyb = full.copy()
        hyb[:, diag, diag] = -np.inf
        hyb = np.where(elim[:, None, :], -np.inf, hyb)            # eliminated opponents do not count
        score = np.where(elim, np.inf, hyb.max(axis=2))
        out.append(risk[np.argmin(score, axis=1)])
        surv = (~elim).sum(axis=1).mean() if surv is None else surv
    return risk, (risk[i_erm], risk[i_tr], out[0], out[1]), surv


# ---------------------------------------------------------------- S2, S3
def trap_class(H, n_each=8):
    """Teacher takes root action 1 w.p. 5%. 'Good' candidates almost never take it and otherwise follow the
    teacher; 'trap' candidates match the root but behave badly inside the rarely shared subtree."""
    base = [0.5 * rng.normal(size=2**h) for h in range(H)]
    def make(root_p1, tilt, sub_shift):
        act = []
        for h in range(H):
            logit = base[h] + 2.0 * tilt * rng.choice([-1.0, 1.0], size=2**h)
            if h >= 1 and sub_shift:
                half = 2**h // 2
                logit[half:] = logit[half:] + sub_shift * rng.choice([-1.0, 1.0], size=half)
            p1 = 1.0 / (1.0 + np.exp(-logit))
            if h == 0:
                p1 = np.array([root_p1])
            act.append(np.stack([1.0 - p1, p1], axis=1))
        return Policy(act)
    t = make(0.05, 0.0, 0.0)
    good = [make(0.001, r, 0.0) for r in np.geomspace(0.01, 0.2, n_each)]
    trap = [make(0.05, r, 2.5) for r in np.geomspace(0.01, 0.2, n_each)]
    return t, good + trap


def check_trap(Hs=(4, 8), sigmas=(0.0, 0.01, 0.03, 0.1, 0.3), trials=1000):
    print("S2/S3. trap instance (8 good + 8 trap candidates); mean risk / L* (se); last column = mean number"
          " of candidates surviving plug-in elimination")
    for H in Hs:
        t, pols = trap_class(H)
        w = design(pols)
        B = max(pathwise_max(p, t) for p in pols)
        first = True
        for s in sigmas:
            risk, sel, surv = run_selectors(pols, t, s, trials if s > 0 else 1, w)
            Ls = risk.min()
            if first:
                print(f"   H={H}: L* = {Ls:.4f} (good family); best trap candidate has risk {risk[8:].min() / Ls:.1f} x L*;"
                      f" B = {B:.2f}")
                print("        sigma   plug-in        tournament     hybrid         hybrid z=1     survivors (hybrid)")
                first = False
            print(f"      {s:7.3f}   " + "   ".join(
                f"{x.mean() / Ls:5.2f} ({x.std() / np.sqrt(len(x)) / Ls:4.2f})" for x in sel) + f"   {surv:6.1f}")


def misspecified_class(H, n_each=32):
    n_states = [3**h for h in range(H)]
    g = [0.5 * rng.normal(size=n) for n in n_states]
    tail = [rng.uniform(0.005, 0.015, size=n) for n in n_states]
    def pol(rho, tail_mass):
        act = []
        for h in range(H):
            p0 = 1.0 / (1.0 + np.exp(-(g[h] + rho * rng.choice([-1.0, 1.0], size=n_states[h]))))
            tm = tail_mass[h] if tail_mass is not None else np.full(n_states[h], 1e-6)
            act.append(np.stack([p0 * (1 - tm), (1 - p0) * (1 - tm), tm], axis=1))
        return Policy(act)
    t = pol(0.0, tail)
    return t, [pol(r, None) for r in np.geomspace(0.02, 0.5, n_each)] + \
              [pol(r, tail) for r in np.geomspace(0.6, 1.2, n_each)]


def check_hybrid_elsewhere(trials=1000):
    print("S3b. hybrid on the earlier instances; mean risk / sigma^2 (realizable) or / L* (misspecified)")
    print("   realizable 64-policy class, sigma = 0.01:   H   plug-in   tournament   hybrid   hybrid z=1   survivors")
    for H in (4, 8, 12):
        base = [0.5 * rng.normal(size=2**h) for h in range(H)]
        t = binary_policy(base, 0.0)
        pols = [t] + [binary_policy(base, r) for r in np.geomspace(0.003, 0.3, 63)]
        risk, sel, surv = run_selectors(pols, t, 0.01, trials, design(pols))
        print(f"                                              {H:3d}  " + "  ".join(f"{x.mean() / 1e-4:8.2f}" for x in sel)
              + f"   {surv:6.1f}")
    print("   misspecified ternary class, H = 6:      sigma   plug-in   tournament   hybrid   hybrid z=1   survivors")
    t, pols = misspecified_class(6)
    w = design(pols)
    for s in (0.01, 0.03, 0.1, 0.3):
        risk, sel, surv = run_selectors(pols, t, s, trials, w)
        print(f"                                           {s:6.2f}  " + "  ".join(f"{x.mean() / risk.min():8.2f}" for x in sel)
              + f"   {surv:6.1f}")


# ---------------------------------------------------------------- S4
def gamma_T(C, S_tilde, w):
    return float(((C**2 @ (1.0 / w)) / S_tilde).max())


def optimal_design(C, S_tilde, w0, iters=400, eta=0.5):
    """min_w max_pairs sum_z C_p(z)^2 / (w(z) S_p): multiplicative weights on pairs, best response in w
    (for fixed pair weights lam, the best w is proportional to sqrt(sum_p lam_p C_p^2 / S_p))."""
    A = C**2 / S_tilde[:, None]
    lam = np.full(A.shape[0], 1.0 / A.shape[0])
    best_w, best = w0, gamma_T(C, S_tilde, w0)
    w_avg = np.zeros_like(w0)
    for it in range(iters):
        b = np.sqrt(lam @ A) + 1e-300
        w = b / b.sum()
        vals = A @ (1.0 / w)
        if vals.max() < best:
            best, best_w = float(vals.max()), w
        w_avg += w
        wa = w_avg / (it + 1)
        va = float((A @ (1.0 / wa)).max())
        if va < best:
            best, best_w = va, wa
        lam = lam * np.exp(eta * vals / vals.max()); lam /= lam.sum()
    return best_w, best


def check_design(n_pol=64):
    print("S4. public design quantity Gamma_T(w) (replaces kappa*H in Theorem 14)")
    print("     class          H   kappa*H (mixture)   Gamma_T(mixture)   Gamma_T(optimal)   single-pair lower bound"
          "   Gamma_T(opt)/H")
    cases = [("realizable", H) for H in (2, 4, 6, 8, 10, 12)] + [("misspecified", H) for H in (2, 4, 6, 8)]
    for kind, H in cases:
        if kind == "realizable":
            base = [0.5 * rng.normal(size=2**h) for h in range(H)]
            t = binary_policy(base, 0.0)
            pols = [t] + [binary_policy(base, r) for r in np.geomspace(0.003, 0.3, n_pol - 1)]
        else:
            t, pols = misspecified_class(H)
        w = design(pols)
        T, pairs, C, const, kap = pair_tables(pols, t, w)
        S = []
        for i, j in pairs:
            m = np.minimum(pols[i].dS, pols[j].dS)
            S.append(m @ np.sum((pols[i].A - pols[j].A) ** 2 / np.maximum(pols[i].A, pols[j].A), axis=1))
        S = np.array(S)
        lower = float((np.abs(C).sum(axis=1) ** 2 / S).max())
        w_opt, g_opt = optimal_design(C, S, w)
        print(f"     {kind:12s} {H:3d}   {kap * H:14.2f}   {gamma_T(C, S, w):14.2f}   {g_opt:14.2f}   {lower:20.2f}"
              f"   {g_opt / H:12.2f}")


if __name__ == "__main__":
    check_proof_chain()
    check_trap()
    check_hybrid_elsewhere()
    check_design()
