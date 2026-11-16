"""Checks for theory.md, section 11 (shared-mass tournament).

Tree MDP with branching k, state = action prefix, public dynamics. Release channel as before:
y(z) = log pi*(z) + N(0, sigma^2 / w(z)), z = (h, s, a), w a public probability over z.

Shared-mass truncated risk:   T_{pi'}(pi) = sum_s min(d^pi(s), d^pi'(s)) * KL(pi_s || pi*_s).
Pairwise statistic:           Delta(pi, pi') = T_{pi'}(pi) - T_{pi}(pi')
                              = sum_s min(d^pi, d^pi')(s) * [KL(pi_s||pi*_s) - KL(pi'_s||pi*_s)],
a same-state comparison on the shared occupancy; it is linear in log pi*, so it is estimated from y.
Tournament:                   pi_hat = argmin_pi max_{pi'} Delta_hat(pi, pi').

Checks
  T1. Transfer lemma (Lemma 12): L(pi) <= (2 + 79 B) T_{pi'}(pi) + 56 B L(pi'), B = max pathwise KL sum,
      on random classes; also the intermediate bound L(pi) <= 2 T + B * P_pi(shared mass ratio drops below 1/2).
  T2. Variance bound (Lemma 13): Var[Delta_hat] <= 8 kappa H sigma^2 (T_{pi'}(pi) + T_{pi}(pi')).
  T3. Realizable class: risk vs H for plug-in, log-space least squares, tournament.
  T4. Misspecified class (as R3 of check_review.py): same three selectors.
"""
import numpy as np

rng = np.random.default_rng(20261115)


class Policy:
    """Per-level action probabilities act[h] of shape (k^h, k); everything else derived."""
    def __init__(self, act):
        self.act = act
        self.H = len(act)
        d = [np.ones(1)]
        for h in range(self.H - 1):
            d.append((d[h][:, None] * act[h]).ravel())
        self.d = d
        self.dS = np.concatenate(d)                                   # occupancy over states, all levels
        self.A = np.concatenate(act, axis=0)                          # (n_states, k)
        self.nu = (self.dS[:, None] * self.A).ravel()
        self.logp = np.log(self.A).ravel()
        self.ent = np.sum(self.A * np.log(self.A), axis=1)            # sum_a pi log pi per state


def local_kl(p, t):
    return np.sum(p.A * (np.log(p.A) - np.log(t.A)), axis=1)          # KL(pi_s || pi*_s) per state


def pathwise_max(p, t):
    """max over trajectories of sum_h KL(pi_{s_h} || pi*_{s_h})."""
    k = p.act[0].shape[1]
    kl_levels, cum, off = [], np.zeros(1), 0
    for h in range(p.H):
        n = p.act[h].shape[0]
        kl_levels.append(np.sum(p.act[h] * (np.log(p.act[h]) - np.log(t.act[h])), axis=1))
    for h in range(p.H):
        cum = cum + kl_levels[h]
        if h < p.H - 1:
            cum = np.repeat(cum, k)
    return float(cum.max())


def crossing_prob(p, q):
    """P_pi(some state on the path has d^q(s) < d^pi(s) / 2)."""
    k = p.act[0].shape[1]
    alive = np.ones(1, dtype=bool)                                    # no crossing so far
    prob = 0.0
    for h in range(p.H):
        cross = alive & (q.d[h] < 0.5 * p.d[h])
        prob += float(p.d[h][cross].sum())
        alive = alive & ~cross
        if h < p.H - 1:
            alive = np.repeat(alive, k)
    return prob


def design(pols):
    H = pols[0].H
    parts = []
    for h in range(H):
        d_mix = sum(p.d[h] for p in pols) / len(pols)
        q_bar = sum(p.act[h] for p in pols) / len(pols)
        parts.append((d_mix[:, None] * q_bar).ravel())
    return np.concatenate(parts) / H


def pair_tables(pols, t, w):
    """Exact T matrix, pairwise coefficient vectors, constants, and the public coverage constant kappa."""
    n, H = len(pols), pols[0].H
    kl = [local_kl(p, t) for p in pols]
    T = np.zeros((n, n))
    pairs, C, const, kap = [], [], [], 0.0
    for i in range(n):
        for j in range(i + 1, n):
            m = np.minimum(pols[i].dS, pols[j].dS)
            T[i, j] = m @ kl[i]; T[j, i] = m @ kl[j]
            C.append((m[:, None] * (pols[i].A - pols[j].A)).ravel())
            const.append(m @ (pols[i].ent - pols[j].ent))
            kap = max(kap, float(((m[:, None] * np.maximum(pols[i].A, pols[j].A)).ravel() / (H * w)).max()))
            pairs.append((i, j))
    return T, np.array(pairs), np.array(C), np.array(const), kap


def tournament(y, pairs, C, const, n):
    """Delta_hat[i, j] estimates T_j(pi_i) - T_i(pi_j); select argmin_i max_j Delta_hat[i, j]."""
    D = const[None, :] - y @ C.T                                      # (trials, n_pairs)
    full = np.zeros((y.shape[0], n, n))
    full[:, pairs[:, 0], pairs[:, 1]] = D
    full[:, pairs[:, 1], pairs[:, 0]] = -D
    full[:, np.arange(n), np.arange(n)] = -np.inf
    return np.argmin(full.max(axis=2), axis=1)


def select_all(pols, t, sigma, trials, w=None):
    n = len(pols)
    w = design(pols) if w is None else w
    NU = np.array([p.nu for p in pols]); LP = np.array([p.logp for p in pols])
    risk = np.sum(NU * (LP - t.logp), axis=1)
    T, pairs, C, const, kap = pair_tables(pols, t, w)
    y = t.logp + rng.normal(size=(trials, w.size)) * sigma / np.sqrt(w)
    i_erm = np.argmin(np.sum(NU * LP, axis=1)[None, :] - y @ NU.T, axis=1)
    i_ls = np.argmin(np.sum(w * LP**2, axis=1)[None, :] - 2.0 * y @ (LP * w).T, axis=1)
    i_tr = tournament(y, pairs, C, const, n)
    return risk, (risk[i_erm], risk[i_ls], risk[i_tr]), kap, T


def binary_policy(base, rho):
    act = []
    for b in base:
        p1 = 1.0 / (1.0 + np.exp(-(b + 2.0 * rho * rng.choice([-1.0, 1.0], size=b.size))))
        act.append(np.stack([1.0 - p1, p1], axis=1))
    return Policy(act)


# ---------------------------------------------------------------- T1, T2
def check_lemmas(H=7, n_pol=14, reps=12, sigma=0.05):
    worst_lemma, worst_mid, worst_var = 0.0, 0.0, 0.0
    for rep in range(reps):
        base = [rng.normal(size=2**h) * rng.choice([0.3, 1.0, 2.0]) for h in range(H)]
        t = binary_policy(base, 0.0)
        spread = rng.choice([0.05, 0.3, 1.0, 2.5])
        pols = [binary_policy(base, spread * rng.uniform(0.1, 1.0)) for _ in range(n_pol)]
        if rep % 3 == 0:                                              # unrelated policies as well
            pols += [binary_policy([rng.normal(size=2**h) * 1.5 for h in range(H)], 0.0) for _ in range(4)]
        w = design(pols)
        L = np.array([np.sum(p.nu * (p.logp - t.logp)) for p in pols])
        B = max(pathwise_max(p, t) for p in pols)
        T, pairs, C, const, kap = pair_tables(pols, t, w)
        var = sigma**2 * (C**2 @ (1.0 / w))
        for (i, j), v in zip(pairs, var):
            worst_var = max(worst_var, v / (8 * kap * H * sigma**2 * (T[i, j] + T[j, i]) + 1e-300))
            for a, b in ((i, j), (j, i)):
                worst_lemma = max(worst_lemma, L[a] / ((2 + 79 * B) * T[a, b] + 56 * B * L[b] + 1e-300))
                worst_mid = max(worst_mid, L[a] / (2 * T[a, b] + B * crossing_prob(pols[a], pols[b]) + 1e-300))
    print(f"T1. transfer lemma over {reps} random classes (binary tree, H={H}): max LHS/RHS = {worst_lemma:.4f};"
          f" intermediate bound L <= 2T + B*P(crossing): max LHS/RHS = {worst_mid:.4f}  (both must be <= 1)")
    print(f"T2. variance bound: max Var / (8 kappa H sigma^2 (T + T')) = {worst_var:.4f}  (must be <= 1)")


# ---------------------------------------------------------------- T3
def check_realizable(sigma=0.01, n_pol=64, trials=1000, Hs=(2, 4, 6, 8, 10, 12)):
    print(f"T3. realizable class ({n_pol} policies, teacher + tilts), sigma = {sigma}; mean risk / sigma^2 (se)")
    print("     H    plug-in          least squares    tournament       kappa_pair   max pathwise KL")
    rows = []
    for H in Hs:
        base = [0.5 * rng.normal(size=2**h) for h in range(H)]
        t = binary_policy(base, 0.0)
        pols = [t] + [binary_policy(base, r) for r in np.geomspace(0.003, 0.3, n_pol - 1)]
        risk, sel, kap, _ = select_all(pols, t, sigma, trials)
        B = max(pathwise_max(p, t) for p in pols)
        r = [(s.mean() / sigma**2, s.std() / np.sqrt(trials) / sigma**2) for s in sel]
        rows.append([x[0] for x in r])
        print(f"   {H:3d}   " + "   ".join(f"{m:7.2f} ({e:4.2f})" for m, e in r) + f"   {kap:8.2f}   {B:10.3f}")
    rows = np.array(rows); lh = np.log(np.array(Hs, dtype=float))
    sl = [np.polyfit(lh[1:], np.log(rows[1:, j]), 1)[0] for j in range(3)]
    print(f"   log-log slope in H (H >= 4): plug-in {sl[0]:.2f}, least squares {sl[1]:.2f}, tournament {sl[2]:.2f}")


# ---------------------------------------------------------------- T4
def check_misspecified(Hs=(2, 4, 6, 8), sigmas=(0.0, 0.01, 0.03, 0.1, 0.3), trials=1000, n_each=32):
    print("T4. misspecified teacher (construction of check_review.py R3); mean risk / L*  (se)")
    for H in Hs:
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
        pols = [pol(r, None) for r in np.geomspace(0.02, 0.5, n_each)] + \
               [pol(r, tail) for r in np.geomspace(0.6, 1.2, n_each)]
        B = max(pathwise_max(p, t) for p in pols)
        first = True
        for s in sigmas:
            risk, sel, kap, T = select_all(pols, t, s, trials if s > 0 else 1)
            Ls = risk.min()
            if first:
                print(f"   H={H}: L* = {Ls:.4f}, max pathwise KL B = {B:.2f}, kappa_pair = {kap:.1f}")
                print("        sigma   plug-in        least squares  tournament")
                first = False
            print(f"      {s:7.3f}   " + "   ".join(
                f"{x.mean() / Ls:5.2f} ({x.std() / np.sqrt(len(x)) / Ls:4.2f})" for x in sel))


if __name__ == "__main__":
    check_lemmas()
    check_realizable()
    check_misspecified()
