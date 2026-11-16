"""Checks added after the 13 Nov 2026 external review of theory.md (section 10).

  R1. Reviewer's rare-action counterexample to the first-pass Corollary 6, and how the sampled estimator's
      convergence scales with the rarity R (corrected statement: the threshold needs a log-ratio second moment).
  R2. Normalization: for normalized teachers, the feasible first-order perturbations of the cross-entropy
      functional <pi, log p*> have size ~ Delta * min_lambda ||pi - lambda p*||_1, not Delta * ||pi||_1.
  R3. Misspecification: teacher outside a public class that contains "tail-ignoring" and "head-distorting"
      candidates; plug-in reverse KL vs log-space least squares under the same Gaussian release channel.
  R4. Growth of the coverage constant kappa with H.

All risks are exact given the selected policy; selection is simulated (R1, R3) with standard errors.
"""
import numpy as np
from scipy.optimize import linprog
from _checks import Checker

chk = Checker("check_review")

rng = np.random.default_rng(20261114)


# ---------------------------------------------------------------- R1
def rare_action(R, T, sigma, trials=200_000):
    """Teacher p* = (e^-R, 1 - e^-R); candidates A = q = (1/R, 1 - 1/R), B = (0, 1); sampling law q."""
    l0, l1 = -float(R), float(np.log1p(-np.exp(-float(R))))
    q0, q1 = 1.0 / R, 1.0 - 1.0 / R
    lq0, lq1 = np.log(q0), np.log1p(-1.0 / R)
    risk_A = q0 * (lq0 - l0) + q1 * (lq1 - l1)
    risk_B = -l1
    # sampled, centred at log q
    n0 = rng.binomial(T, q0, size=trials) / T
    n1 = 1.0 - n0
    z0 = rng.normal(size=trials) * sigma * np.sqrt(n0)
    z1 = rng.normal(size=trials) * sigma * np.sqrt(n1)
    corr0, corr1 = n0 * (l0 - lq0) + z0, n1 * (l1 - lq1) + z1
    est_A = -(corr0 + corr1)                       # KL(A || q) = 0, ratios A/q = 1
    est_B = -lq1 - corr1 / q1                      # entropy term 0, ratio B/q = (0, 1/q1)
    pick_A = est_A < est_B
    r_s = np.where(pick_A, risk_A, risk_B)
    # weighted full release with precision ∝ q
    y0 = l0 + rng.normal(size=trials) * sigma / np.sqrt(q0)
    y1 = l1 + rng.normal(size=trials) * sigma / np.sqrt(q1)
    est_Af = q0 * lq0 + q1 * lq1 - (q0 * y0 + q1 * y1)
    est_Bf = -y1
    r_f = np.where(est_Af < est_Bf, risk_A, risk_B)
    return r_s.mean(), r_f.mean(), (1 - q0) ** T, risk_A


def check_rare_action(sigma=0.02):
    print("R1. rare-action instance, Delta/mu = 0.02 (first-pass T* = G mu^2 / (e^b Delta^2) ≈ 2500 for every R)")
    print("       R          T    sampled risk   P(rare action unseen)   weighted-full risk")
    for R, Ts in [(10**6, (10**5, 10**6, 10**7)), (10**4, (10**3, 10**4, 10**5)), (10**2, (10, 10**2, 10**3))]:
        for T in Ts:
            rs, rf, p0, rA = rare_action(R, T, sigma)
            chk.close(f"R1 sampled risk = P(unseen) * risk(A), R={R}, T={T}", rs, p0 * rA, rel=0.0, abs_tol=0.01)
            chk.le(f"R1 weighted-full risk ~ 0, R={R}, T={T}", rf, 1e-3)
            print(f"   {R:8.0e}  {T:9.0e}   {rs:10.4f}      {p0:10.4f}            {rf:10.6f}")
    print("   -> sampled feedback converges only once T is a few multiples of R; no threshold in (G, e^b, Delta/mu) alone.")


# ---------------------------------------------------------------- R2
def check_normalization(d=8, Delta=0.05, trials=200):
    """max <c, delta> over the linearized feasible set {|delta_a| <= Delta, <p*, delta> = 0}."""
    def width(c, p):
        res = linprog(-c, A_eq=p[None, :], b_eq=[0.0], bounds=[(-Delta, Delta)] * len(c), method="highs")
        return -res.fun
    r_pos, r_diff, r_near = [], [], []
    for _ in range(trials):
        p = rng.dirichlet(np.ones(d))
        pi = rng.dirichlet(np.ones(d)); pi2 = rng.dirichlet(np.ones(d))
        near = 0.95 * p + 0.05 * pi
        r_pos.append(width(pi, p) / Delta)                       # cube calibration would say ||pi||_1 = 1
        r_diff.append(width(pi - pi2, p) / (Delta * np.abs(pi - pi2).sum()))
        r_near.append(width(near, p) / Delta)
    chk.le("R2 zero-sum width at least half the cube value", 0.5, np.min(r_diff))
    chk.le("R2 width never exceeds the cube value", max(np.max(r_pos), np.max(r_diff), np.max(r_near)), 1.0, tol=1e-7)
    print("R2. feasible first-order width of <c, log p*> for normalized teachers, relative to the cube value Delta*||c||_1")
    print(f"   c = a random policy           : mean {np.mean(r_pos):.3f}  (cube calibration assumes 1)")
    print(f"   c = policy 5% away from p*    : mean {np.mean(r_near):.3f}")
    print(f"   c = difference of two policies: mean {np.mean(r_diff):.3f}, min {np.min(r_diff):.3f}  (two-point bound uses >= 1/2)")


# ---------------------------------------------------------------- tree helpers (branching k)
def occupancy(act):
    d = [np.ones(1)]
    for h in range(len(act) - 1):
        d.append((d[h][:, None] * act[h]).ravel())
    nu = np.concatenate([(dh[:, None] * a).ravel() for dh, a in zip(d, act)])
    logp = np.concatenate([np.log(a).ravel() for a in act])
    return d, nu, logp


def design(acts, occs, H):
    n = len(acts)
    parts = []
    for h in range(H):
        d_mix = sum(o[0][h] for o in occs) / n
        q_bar = sum(a[h] for a in acts) / n
        parts.append((d_mix[:, None] * q_bar).ravel())
    return np.concatenate(parts) / H


def kappa(acts, occs, t_act, w, H, idx):
    """max_z d^pi(s) max(pi, pi*)(a|s) / (H w(z)) for the policies in idx."""
    out = []
    for i in idx:
        v = np.concatenate([(dh[:, None] * np.maximum(a, ta)).ravel()
                            for dh, a, ta in zip(occs[i][0], acts[i], t_act)])
        out.append(float((v / (H * w)).max()))
    return np.array(out)


def softmax_rows(g):
    e = np.exp(g - g.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- R3
def check_misspecified(Hs=(2, 4, 6, 8), sigmas=(0.0, 0.01, 0.03, 0.1, 0.3), trials=1000, n_each=32):
    """Teacher: 2-action head plus a tail action of mass 0.5-1.5% at every state; it is not in the class.
    Class (public): 32 tail-ignoring candidates (tail mass 1e-6, head logit shifted by ±rho, rho in
    [0.02, 0.5]) and 32 head-distorting candidates (exact tail, head logit shifted by ±rho, rho in [0.6, 1.2]).
    Built so that reverse KL prefers a tail-ignoring candidate and log-space L2 prefers a head-distorting one."""
    print("R3. misspecified teacher; reverse KL and log-space L2 rank the candidates differently")
    kinds = np.array(["tail-ignoring"] * n_each + ["head-distorting"] * n_each)
    for H in Hs:
        n_states = [3**h for h in range(H)]
        g = [0.5 * rng.normal(size=n) for n in n_states]              # head logit of action 0 vs 1
        tail = [rng.uniform(0.005, 0.015, size=n) for n in n_states]
        def pol(rho, tail_mass):
            act = []
            for h in range(H):
                p0 = 1.0 / (1.0 + np.exp(-(g[h] + rho * rng.choice([-1.0, 1.0], size=n_states[h]))))
                tm = tail_mass[h] if tail_mass is not None else np.full(n_states[h], 1e-6)
                act.append(np.stack([p0 * (1 - tm), (1 - p0) * (1 - tm), tm], axis=1))
            return act
        t_act = pol(0.0, tail)
        acts = [pol(r, None) for r in np.geomspace(0.02, 0.5, n_each)] + \
               [pol(r, tail) for r in np.geomspace(0.6, 1.2, n_each)]
        t_occ = occupancy(t_act)
        occs = [occupancy(a) for a in acts]
        NU = np.array([o[1] for o in occs]); LP = np.array([o[2] for o in occs])
        w = design(acts, occs, H)
        risk = np.sum(NU * (LP - t_occ[2]), axis=1)
        Lstar = risk.min()
        Dw = np.sum(w * (LP - t_occ[2]) ** 2, axis=1)
        i_kl, i_ls = int(np.argmin(risk)), int(np.argmin(Dw))
        kap = kappa(acts, occs, t_act, w, H, range(len(acts)))
        print(f"   H={H}: L* = {Lstar:.4f} ({kinds[i_kl]}); noiseless LS picks a {kinds[i_ls]} candidate with risk"
              f" {risk[i_ls]:.4f} = {risk[i_ls] / Lstar:.1f} x L*; kappa median {np.median(kap):.1f}, max {kap.max():.1f}")
        print("        sigma   plug-in risk (se)    LS risk (se)        plug-in/L*   LS/L*   LS picks tail-ignoring")
        for s in sigmas:
            y = t_occ[2] + rng.normal(size=(trials, w.size)) * s / np.sqrt(w)
            erm = np.sum(NU * LP, axis=1)[None, :] - y @ NU.T
            ls = np.sum(w * LP**2, axis=1)[None, :] - 2.0 * y @ (LP * w).T
            ie, il = chk.argmin('R3 plug-in scores', erm), chk.argmin('R3 least-squares scores', ls)
            re, rl = risk[ie], risk[il]
            chk.finite(f"R3 risks H={H} sigma={s}", [re.mean(), rl.mean()])
            if s == 0.0:
                chk.close(f"R3 noiseless plug-in returns L*, H={H}", re.mean(), Lstar, rel=1e-9)
            print(f"      {s:7.3f}   {re.mean():.4f} ({re.std() / np.sqrt(trials):.4f})   "
                  f"{rl.mean():.4f} ({rl.std() / np.sqrt(trials):.4f})   "
                  f"{re.mean() / Lstar:9.2f}   {rl.mean() / Lstar:6.2f}   {(il < n_each).mean():10.2f}")


# ---------------------------------------------------------------- R4
def check_kappa(Hs=(2, 4, 6, 8, 10, 12), n_pol=64):
    print("R4. coverage constant kappa for the realizable 64-policy class of check_horizon.py (binary tree,"
          " design d_mix ⊗ q_bar)")
    print("     H   kappa: teacher   median over class   max over class   max over 8 nearest candidates")
    for H in Hs:
        base = [0.5 * rng.normal(size=2**h) for h in range(H)]
        def pol(rho):
            act = []
            for h in range(H):
                p1 = 1.0 / (1.0 + np.exp(-(base[h] + 2.0 * rho * rng.choice([-1.0, 1.0], size=2**h))))
                act.append(np.stack([1.0 - p1, p1], axis=1))
            return act
        acts = [pol(0.0)] + [pol(r) for r in np.geomspace(0.003, 0.3, n_pol - 1)]
        occs = [occupancy(a) for a in acts]
        w = design(acts, occs, H)
        kap = kappa(acts, occs, acts[0], w, H, range(n_pol))
        chk.finite(f"R4 kappa H={H}", kap)
        chk.le(f"R4 kappa >= 1 for the teacher, H={H}", 1.0, kap[0], tol=1e-9)
        print(f"   {H:3d}   {kap[0]:12.2f}   {np.median(kap):15.2f}   {kap.max():13.2f}   {kap[1:9].max():16.2f}")


if __name__ == "__main__":
    check_rare_action()
    check_normalization()
    check_misspecified()
    check_kappa()
    chk.finish()
