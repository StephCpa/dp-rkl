"""Checks for theory.md, section 14 (answer to the re-review of PR #1).

  V1. Envelope mass. The claim E <= H min{d, |Pi|} is false on multi-step trees; the correct general bounds
      are E <= sum_h min{|Pi|, d^h} <= H |Pi|. A full-support counterexample with H = 2, d = 2, |Pi| = 4.
  V2. Near-duplicate pairs. The design ratio is scale-invariant, so a nearly duplicate pair must be kept.
      An instance where dropping it (the old S > 1e-14 filter) makes a design look good while its true
      Gamma_T is enormous; and a class of exact duplicates, which must not raise.
  V3. Negative test of the failure detection: NaN noise must make the Checker fail, not pass silently.
  V4. Anchor bias. A realizable class, zero noise, anchor = candidate mean: the anchor-normalized plug-in
      selects a wrong candidate, as Proposition 16's A_q term allows.
  V5. H x anchor-quality sweep for the anchor-normalized plug-in (Proposition 16 estimator, product design
      omega(s) q(a|s) with the same anchor q), with the plug-in and the tournament for reference.
      The class seed is the statistical unit; slopes are per-seed log-log fits with a 95% t-interval.
"""
import numpy as np
import check_tournament as ct
import check_round2 as r2
from check_tournament import Policy
from _checks import Checker
from _pairs import pair_struct, gamma_T, optimal_design

chk = Checker("check_round3")


# ---------------------------------------------------------------- V1
def check_envelope():
    eps = 0.02
    def pol(a1, a2_left, a2_right):                      # binary tree, H = 2; probabilities of action 1
        root = np.array([[1 - a1, a1]])
        lvl2 = np.array([[1 - a2_left, a2_left], [1 - a2_right, a2_right]])
        return Policy([root, lvl2])
    pols = [pol(eps, eps, 0.5), pol(eps, 1 - eps, 0.5), pol(1 - eps, 0.5, eps), pol(1 - eps, 0.5, 1 - eps)]
    NU = np.array([p.nu for p in pols])
    E = float(NU.max(axis=0).sum())
    H, d, n = 2, 2, len(pols)
    wrong = H * min(d, n)
    right = sum(min(n, d**h) for h in range(1, H + 1))
    print(f"V1. H = 2, d = 2, |Pi| = 4, full support (min prob {NU.min():.4f}): envelope mass E = {E:.4f};"
          f" withdrawn bound H*min(d,|Pi|) = {wrong}; correct bound sum_h min(|Pi|, d^h) = {right} <= H|Pi| = {H * n}")
    chk.le("V1 the withdrawn bound is violated", wrong, E - 1e-6)
    chk.le("V1 E <= sum_h min(|Pi|, d^h)", E, right)
    chk.le("V1 E >= H", H, E)
    _, _, _, Cn, Sn, valid = pair_struct(pols)
    w_env = NU.max(axis=0) / E
    chk.le("V1 Gamma_T(envelope design) <= E", gamma_T(Cn, Sn, valid, w_env), E, tol=1e-9)


# ---------------------------------------------------------------- V2
def check_near_duplicates():
    base = np.array([0.4, 0.3, 0.2, 0.1])
    delta = 1e-8
    near = base + np.array([0.0, 0.0, delta, -delta])            # differs from base by 1e-8 on two actions
    far = np.array([0.1, 0.6, 0.2, 0.1])                         # differs on the first two actions only
    pols = [Policy([v[None, :]]) for v in (base, near, far)]
    pairs, C, const, Cn, Sn, valid = pair_struct(pols)
    # what the old filter did: keep a pair only if its unnormalized S exceeds 1e-14
    S_raw = np.array([np.sum((pols[i].A - pols[j].A) ** 2 / np.maximum(pols[i].A, pols[j].A)) for i, j in pairs])
    old_keep = S_raw > 1e-14
    w_old, ub_old, _ = optimal_design(Cn, Sn, old_keep, np.full(4, 0.25), iters=2000)
    true_at_old = gamma_T(Cn, Sn, valid, w_old)
    w_new, ub_new, lb_new = optimal_design(Cn, Sn, valid, np.full(4, 0.25), iters=2000)
    print(f"V2. near-duplicate pair (difference 1e-8): raw S = {S_raw.min():.2e}, dropped by the old filter:"
          f" {int((~old_keep).sum())} of {len(pairs)} pairs")
    print(f"    design optimized without it: reported {ub_old:.4f}, but its Gamma_T over all pairs = {true_at_old:.3e}")
    print(f"    design optimized with all pairs: Gamma_T = {ub_new:.4f}, certified lower bound {lb_new:.4f}")
    chk.le("V2 all three pairs are valid", 3, int(valid.sum()))
    chk.le("V2 the old filter drops the near-duplicate pair", 1, int((~old_keep).sum()))
    chk.le("V2 old design is bad once all pairs count", 1e3, true_at_old)
    chk.le("V2 new optimum is certified", lb_new, ub_new, tol=1e-6 * ub_new)
    chk.le("V2 new optimum <= envelope mass", lb_new, float(np.array([p.nu for p in pols]).max(axis=0).sum()))
    chk.close("V2 reported value equals the full definition for the new design", gamma_T(Cn, Sn, valid, w_new),
              ub_new, rel=1e-9)
    # exact duplicates: no valid pair, Gamma_T = 0 by definition, nothing raises
    dup = [Policy([base[None, :]]) for _ in range(3)]
    _, _, _, Cn, Sn, valid = pair_struct(dup)
    w, ub, lb = optimal_design(Cn, Sn, valid, np.full(4, 0.25))
    print(f"    three exact duplicates: valid pairs = {int(valid.sum())}, Gamma_T = {gamma_T(Cn, Sn, valid, w)}")
    chk.close("V2 exact duplicates give Gamma_T = 0", gamma_T(Cn, Sn, valid, w), 0.0, rel=0, abs_tol=0)
    chk.finite("V2 fallback design", w)


# ---------------------------------------------------------------- V3
def check_nan_detection():
    t, pols = r2.make_class(("realizable", 4, 8, 0.3), 7)
    pairs, C, const, Cn, Sn, valid = pair_struct(pols)
    w = ct.design(pols)
    probe = Checker("probe")
    g = np.full((5, w.size), np.nan)
    out, _ = r2.selectors(pols, t, 0.01, g, w, pairs, C, const, checker=probe)
    finite_risks = all(np.all(np.isfinite(v)) for v in out.values())
    print(f"V3. NaN noise: selectors still return finite risks ({finite_risks}), which is why risks alone are not"
          f" a check; the Checker recorded {len(probe.failures)} failures on inputs and scores")
    chk.le("V3 NaN input is detected", 1, len(probe.failures))
    clean = Checker("probe-clean")
    r2.selectors(pols, t, 0.01, np.zeros((5, w.size)), w, pairs, C, const, checker=clean)
    chk.close("V3 clean input raises no failure", len(clean.failures), 0, rel=0, abs_tol=0)


# ---------------------------------------------------------------- V4
def check_anchor_bias():
    e = 1e-9
    def pol(root_right, right_b):                    # H = 2 binary tree; left subtree identical for everyone
        return Policy([np.array([[1 - root_right, root_right]]),
                       np.array([[0.5, 0.5], [1 - right_b, right_b]])])
    teacher = pol(0.01, e)                           # rarely goes right; there it plays action a
    pols = [teacher, pol(0.99, e),                   # candidate 1: wrong at the root, right elsewhere
            pol(0.01, 1 - e), pol(0.01, 1 - e)]      # two candidates that play b in the right subtree
    q = np.mean([p.A for p in pols], axis=0)         # anchor = candidate mean
    kl_anchor = np.sum(q * (np.log(q) - np.log(teacher.A)), axis=1)
    L = np.array([np.sum(p.nu * (p.logp - teacher.logp)) for p in pols])
    Cq = np.array([p.dS @ kl_anchor for p in pols])
    J = L - Cq                                       # what the anchor-normalized plug-in minimizes at zero noise
    pick = int(np.argmin(J))
    bound = 3 * L.min() + 4 * Cq.max()
    print(f"V4. realizable, zero noise, anchor = candidate mean: risks {np.round(L, 3)}, anchor terms {np.round(Cq, 3)};"
          f" selected candidate {pick} with risk {L[pick]:.3f}; Proposition 16 bound 3L* + 4A_q = {bound:.2f}")
    chk.close("V4 the class is realizable", L.min(), 0.0, rel=0, abs_tol=1e-12)
    chk.le("V4 a wrong candidate is selected at zero noise", 1.0, L[pick])
    chk.le("V4 the selection obeys Proposition 16", L[pick], bound)


# ---------------------------------------------------------------- V5
def tilt(t, tau, where, rng):
    """Anchor = teacher with logit shifts +-2*tau on all states ('all') or only below root action 1 ('half')."""
    act = []
    for h, a in enumerate(t.act):
        logit = np.log(a[:, 1]) - np.log(a[:, 0])
        shift = 2.0 * tau * rng.choice([-1.0, 1.0], size=logit.size)
        if where == "half":
            mask = np.zeros(logit.size); mask[logit.size // 2:] = 1.0 if h >= 1 else 0.0
            shift = shift * mask
        p1 = 1.0 / (1.0 + np.exp(-(logit + shift)))
        act.append(np.stack([1.0 - p1, p1], axis=1))
    return Policy(act)


def check_sweep(Hs=(2, 4, 6, 8, 10, 12), n_seeds=8, trials=300, sigma=0.01, n_pol=32):
    anchors = [("teacher", 0.0, "all"), ("uniform tilt 0.05", 0.05, "all"), ("uniform tilt 0.15", 0.15, "all"),
               ("uniform tilt 0.4", 0.4, "all"), ("half-tree tilt 0.4", 0.4, "half"),
               ("half-tree tilt 1.5", 1.5, "half"), ("candidate mean", None, None)]
    names = [a[0] for a in anchors] + ["plug-in (ref)", "tournament (ref)"]
    risk = {nm: np.zeros((n_seeds, len(Hs))) for nm in names}
    bias = {nm: np.zeros((n_seeds, len(Hs))) for nm in names}
    Aq = {nm: np.zeros((n_seeds, len(Hs))) for nm in names}
    for si in range(n_seeds):
        for hi, H in enumerate(Hs):
            ct.rng = np.random.default_rng(7000 + 100 * si + H)
            base = [0.5 * ct.rng.normal(size=2**h) for h in range(H)]
            t = ct.binary_policy(base, 0.0)
            pols = [t] + [ct.binary_policy(base, r) for r in np.geomspace(0.003, 0.3, n_pol - 1)]
            n, k = len(pols), 2
            NU = np.array([p.nu for p in pols]); LP = np.array([p.logp for p in pols])
            DS = np.array([p.dS for p in pols])
            L = np.sum(NU * (LP - t.logp), axis=1)
            d_mix = DS.mean(axis=0)
            g = np.random.default_rng(90_000 + 100 * si + H).normal(size=(trials, NU.shape[1]))
            arng = np.random.default_rng(80_000 + 100 * si + H)
            c_erm = np.sum(NU * LP, axis=1)
            for nm, tau, where in anchors:
                q = np.mean([p.A for p in pols], axis=0) if tau is None else tilt(t, tau, where, arng).A
                w = ((d_mix[:, None] * q) / H).ravel()                     # product design with this anchor
                kl_q = np.sum(q * (np.log(q) - np.log(t.A)), axis=1)
                Cq = DS @ kl_q
                NUa = (NU.reshape(n, -1, k) - DS[:, :, None] * q[None]).reshape(n, -1)
                c_a = c_erm - DS @ np.sum(q * np.log(q), axis=1)
                y = t.logp + sigma * g / np.sqrt(w)
                chk.finite("V5 release", y)
                risk[nm][si, hi] = L[chk.argmin("V5 anchor scores", c_a[None, :] - y @ NUa.T)].mean()
                bias[nm][si, hi] = L[int(np.argmin(L - Cq))]              # zero-noise selection
                Aq[nm][si, hi] = Cq.max()
            w = ct.design(pols)
            y = t.logp + sigma * g / np.sqrt(w)
            risk["plug-in (ref)"][si, hi] = L[chk.argmin("V5 plug-in scores", c_erm[None, :] - y @ NU.T)].mean()
            pairs, C, const, _, _, _ = pair_struct(pols)
            D = const[None, :] - y @ C.T
            chk.finite("V5 tournament statistics", D)
            full = np.zeros((trials, n, n))
            full[:, pairs[:, 0], pairs[:, 1]] = D; full[:, pairs[:, 1], pairs[:, 0]] = -D
            full[:, np.arange(n), np.arange(n)] = -np.inf
            risk["tournament (ref)"][si, hi] = L[np.argmin(full.max(axis=2), axis=1)].mean()
    lh = np.log(np.array(Hs, dtype=float))
    t975 = r2.T975[n_seeds - 1]
    print(f"\nV5. H x anchor quality, realizable class ({n_pol} policies), sigma = {sigma}, {n_seeds} class seeds;"
          " risk / sigma^2, mean over seeds")
    print("    anchor                 " + "".join(f"   H={H:<4d}" for H in Hs)
          + "   slope in H [95% interval]   zero-noise risk/sigma^2 at H=12   A_q/sigma^2 at H=12")
    slopes = {}
    for nm in names:
        r = risk[nm] / sigma**2
        sl = np.array([np.polyfit(lh[1:], np.log(r[si, 1:]), 1)[0] for si in range(n_seeds)])
        half = t975 * sl.std(ddof=1) / np.sqrt(n_seeds)
        slopes[nm] = (sl.mean(), sl.mean() - half, sl.mean() + half)
        tail = (f"{bias[nm][:, -1].mean() / sigma**2:14.2f}                 {Aq[nm][:, -1].mean() / sigma**2:12.1f}"
                if nm in Aq and "ref" not in nm else "")
        print(f"    {nm:20s}" + "".join(f"{v:9.2f}" for v in r.mean(axis=0))
              + f"   {sl.mean():5.2f} [{sl.mean() - half:5.2f}, {sl.mean() + half:5.2f}]          {tail}")
    chk.between("V5 slope with the teacher as anchor", slopes["teacher"][0], 0.8, 1.4)
    chk.between("V5 plug-in reference slope", slopes["plug-in (ref)"][0], 1.6, 2.5)
    chk.close("V5 teacher anchor has zero bias", bias["teacher"].max(), 0.0, rel=0, abs_tol=1e-12)


if __name__ == "__main__":
    check_envelope()
    check_near_duplicates()
    check_nan_detection()
    check_anchor_bias()
    check_sweep()
    chk.finish()
