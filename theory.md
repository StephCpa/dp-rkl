# Private feedback design for reverse-KL distillation: theory note (12 Nov 2026)

Builds on Liu, Wei, Zimmert, "Reverse-KL Policy Distillation: Statistical Guarantees and Limits" (arXiv 2610.30844), read through an automated extraction; verify theorem numbers against the PDF before citing. Setting is the private-teacher model of DP-OPD-004 (claude/dp-opd-ier.md). It settles, in a restricted model, conjecture ④ of claude/dp-opsd-feasibility.md. Checks: checks/check_feedback_design.py (§2–4, §8), check_horizon.py (§9), check_review.py (§10), check_tournament.py (§11) and check_hybrid.py (§12).

**Revised 13 Nov 2026 after an external review.** Corollary 6 was wrong and is replaced by Proposition 6′; the lower bounds are restated for normalized teachers; the H² claim is restricted to the plug-in estimator; a misspecification experiment is added. §10 lists every change.

## 0. Results at a glance

| # | Statement | Status |
|---|---|---|
| L1 | A sampled release with law r and a deterministic release with precision shares βr have identical privacy-noise variance; the sampled one adds sampling variance on top | Proved; a variance identity only, it does not give equal risk (see P6′) |
| L2 | For one linear functional c, the best diagonal Gaussian design calibrated to the cube is precision ∝ \|c_a\|, variance Δ²‖c‖₁²/β. A matching worst-case lower bound over all GDP mechanisms holds for zero-sum c (differences of policies) only | Proved; lower bound restated 13 Nov for normalized teachers |
| P3 | For a policy class, the privacy inflation is Γ(Π) with 1 ≤ Γ ≤ min{d, \|Π\|, e^b}, the same three-way minimum as the paper's action-feedback factor | Proved; tight for deterministic classes |
| P4 | Calibrating Gaussian noise to a cube makes full and sampled release tie in the privacy term; calibrating to an ℓ2 ball lets full release win by up to ‖c‖₁²/‖c‖₂² | Proved as a comparison of calibrated Gaussian mechanisms; not a minimax claim |
| T5 | H=1: L(π̂) ≤ 9L* + 11·G·ι/T + 32·e^b·(Δ/μ)²·ι, with ι = log(2\|Π\|/δ); the privacy term does not depend on T | Proved; simulation agrees on shape, constants loose (~150×) |
| P6′ | Sampled feedback has a privacy term of the same order as weighted full release, but its T-term scales with a log-ratio second moment V̄ ≤ 8e^b·Λ_c², not with G. No threshold depending only on (G, e^b, Δ/μ) exists | Proved, constants unoptimized. Replaces the first-pass Corollary 6, which was wrong |
| P7 | Gradient noise: optimal design gives (E_p‖e_a−p‖)², sampled token gives E_p‖e_a−p‖² = 1−‖p‖²; ratio 1+CV² | Proved; Monte Carlo agrees |
| P8 | Lower bounds for proper learners: realizable floor Ω(min{1,(Δ/μ)²}); with C_app = 1 the floor is Ω(min{1, Δ/μ}) | Proved (two-point, normalized teachers, richness assumption); improper learners not written out |
| T9 | H>1, plug-in reverse-KL ERM: L(π̂) ≤ 3L* + 16·e^b·H²·(Δ/μ)²·ι; no clipping needed | Proved (public dynamics). H² observed for this estimator on a tree MDP; it is not a lower bound for other mechanisms |
| T10 | H>1, log-space least squares (the MLE for the release channel): realizable L(π̂) ≤ 8·κ·H·(Δ/μ)²·ι | Proved; slope 1.00 in H on the tree MDP; κ can grow with H for far candidates (§10) |
| P11 | H>1 lower bounds: realizable Ω(H(Δ/μ)²); C_app = 1 gives Ω(HΔ/μ) | Proved (two-point, normalized, proper learners) |
| L12–T14 | H>1, shared-mass tournament: L(π̂) ≤ (14 + 609·B̄)·L* + (64 + 2528·B̄)·Γ_T(w)·(Δ/μ)²·ι, with B̄ the largest pathwise KL sum in the class and Γ_T(w) a public design quantity, about 1.0–1.8·H in the classes tested (§12). Reverse-KL approximation term with a floor linear in H when B̄ is bounded | Proved (own proof, not externally checked; constants crude). Every step of the transfer lemma verified exactly on 800 ordered pairs (§12). The B̄ factor is real: a trap instance makes the tournament select 3.7·L* with no noise |
| P15 | Hybrid (plug-in elimination with publicly known noise levels, then tournament among survivors) satisfies the smaller of the T9-type and T14 bounds | Proved. In simulation it coincides with the tournament unless the noise is near zero, so the practical gain is small |
| §10 | Misspecified teacher: plug-in stays at L* for small noise, least squares sits at about 4·L* at every noise level; least squares wins only once privacy noise dominates | Simulation on one constructed family |

## 1. Model

- Actions A, d = |A|. Teacher log-probabilities ℓ*_s ∈ R^d at each state s, a function of the private dataset D.
- **Sensitivity (S∞).** For neighbouring D, D′: sup_s ‖ℓ*_s(D) − ℓ*_s(D′)‖_∞ ≤ Δ. For the floored product-of-experts teacher over M disjoint shards, Δ = 2Λ/M with Λ = log(1/γ). One shard can move every coordinate at once, so the sensitivity body is a cube, not a ball.
- **Release primitives**, both using only public choices (state, design):
  - *Deterministic design w* (w_a ≥ 0, Σ_a w_a = β): output y_a = ℓ*_a + N(0, Δ²/w_a) for every a with w_a > 0. Rescaling coordinate a by √w_a/Δ gives ℓ2 sensitivity ≤ √β, so the cost is √β-GDP.
  - *Sampled design r*: draw a ~ r, output a and ℓ*_a + N(0, Δ²/β). Cost √β-GDP.
- **Budget.** N positions with shares β_n fixed in advance, Σβ_n = μ². States and designs may depend on earlier outputs (adaptive GDP composition).
- "Full feedback" in the paper = a deterministic design with full support. "Action feedback" = a sampled design with r = the rollout policy. Homoscedastic full release = w uniform.

## 2. Per-position results

**Lemma 1 (design identity).** For c ∈ R^d with supp(c) ⊆ supp(r):
- deterministic, w = βr: ⟨c, y⟩ is unbiased with variance V_det = (Δ²/β)·Σ_a c_a²/r_a;
- sampled: (c_a/r_a)·y is unbiased with variance V_det + Var_{a~r}(c_aℓ*_a/r_a).

Proof: direct computation; the privacy part of the sampled estimator is E_r[(c_a/r_a)²]·Δ²/β. ∎

This is an identity about variances. The sampled estimator's privacy noise is Gaussian only conditionally on the sampled actions (a scale mixture unconditionally), and its sampling error depends on log-ratios at rare actions. Equal privacy variance does not imply equal tails, misselection probability or risk; see Proposition 6′.

**Lemma 2 (one functional).** Among diagonal Gaussian designs calibrated to the cube, min_w V_det = Δ²‖c‖₁²/β, attained at w ∝ |c_a| (Cauchy–Schwarz). Releasing ⟨c, ℓ*⟩ with cube-calibrated noise gives the same variance.

*Lower bound (restated 13 Nov).* Teachers are normalized, so a feasible perturbation δ of the log-probabilities satisfies Σ_a p*_a·e^{δ_a} = 1. It cannot move every coordinate the same way, and the cube is only an outer bound on the perturbation set. Richness assumption R_k: for normalized teachers p_± ∝ p₀·exp(±η·sign(c)) with 4η ≤ kΔ (so ‖log p₊ − log p₋‖_∞ ≤ kΔ) there are datasets at distance ≤ k producing them. If c sums to zero, ⟨c, log p₊ − log p₋⟩ = 2η‖c‖₁. With k = ⌈1/√β⌉ the two datasets are ≤ 2-GDP apart, so any √β-GDP estimator of ⟨c, ℓ*⟩ has worst-case squared error ≥ c₀·Δ²‖c‖₁²/β over {p₊, p₋}. ∎

So for differences of policies, which is what selection needs, the cube-calibrated design is worst-case optimal up to constants. For c with nonzero sum (one policy's cross-entropy, c = π) no such bound holds. To first order feasible perturbations satisfy ⟨p*, δ⟩ = 0 and the width is Δ·min_λ‖c − λp*‖₁. Numerically (check R2) this is 0.76·Δ for a random policy and 0.04·Δ for a policy 5% away from the teacher, against the cube value Δ. Cube-calibrated noise is then conservative, and a normalization-aware mechanism could do better (open problem 2).

**Proposition 3 (classes).** For a set C of vectors let Γ(C) = min_{w ∈ simplex} max_{c ∈ C} Σ_a c_a²/w_a (a convex program). Every c ∈ C then has privacy variance ≤ Δ²Γ(C)/β under the optimal diagonal design. For C a set of probability vectors:
- Γ ≥ 1 (Lemma 2, upper part);
- Γ ≤ d·max‖c‖₂² ≤ d (uniform design);
- Γ ≤ |C| (design = average of C);
- Γ ≤ e^b if c ≤ e^b·q for a public reference q (design = q).

Hence 1 ≤ Γ ≤ min{d, |C|, e^b}, and Γ = d for the class of deterministic policies. The paper's online action-feedback penalty is m = min{|A|, |Π|, e^{B∞}}; here the same minimum bounds the privacy inflation within diagonal designs, with coverage by a *public reference* replacing coverage by the teacher. ∎

**Proposition 4 (geometry of calibration).** Suppose Gaussian noise is calibrated to a symmetric convex outer bound K of the perturbation set, with support function h_K. Releasing ⟨c, ℓ*⟩ directly has noise variance h_K(c)²/β; the best sampled design has (Σ_a |c_a|·h_K(e_a))²/β ≥ h_K(c)²/β. Cube: both equal Δ²‖c‖₁²/β. ℓ2 ball of radius Δ₂: Δ₂²‖c‖₂² against Δ₂²‖c‖₁². This compares calibrated Gaussian mechanisms; it is not a minimax statement over all mechanisms (a constant output has zero variance; worst-case error bounds need a richness assumption as in Lemma 2). The practical reading: whether full release keeps an advantage in the privacy term depends on how sensitivity is bounded, and a designer who clips shard log-ratio vectors in ℓ2 instead of flooring coordinatewise changes the answer.

## 3. Rate theorem, H = 1

**Setting.** Public contexts s_1..s_T i.i.d.; L(π) = E_s KL(π_s‖π*_s); finite Π; pathwise k(s;π) = KL(π_s‖π*_s) ≤ G (G ≤ Λ for a floored teacher); a public design q_s with q_s ≥ e^{−b}·max(π_s, π*_s) for all π ∈ Π. Total budget μ-GDP, β = μ²/T, weighted full release with w_t = β·q_{s_t}. Estimator: L̃(π) = (1/T)Σ_t[Σ_a π log π − ⟨π_{s_t}, y_t⟩], π̂ = argmin L̃.

**Theorem 5.** With probability ≥ 1 − 2δ, writing ι = log(2|Π|/δ),

  L(π̂) ≤ 9·L* + 11·G·ι/T + 32·e^b·(Δ/μ)²·ι.

If π* ∈ Π: L(π̂) ≤ 3·G·ι/T + 16·e^b·(Δ/μ)²·ι.

Proof.
1. L̃(π) = L̂(π) − Z(π) with L̂ the noiseless empirical risk and Z(π) = (1/T)Σ_t⟨π_{s_t}, ξ_t⟩.
2. Var[Z(π) − Z(π°)] = (Δ²/(Tμ²))·Σ_t Σ_a (π−π°)²/q. Using (π−π°)² ≤ 2(π−π*)² + 2(π°−π*)², q ≥ e^{−b}max(·), and Σ_a (p−p′)²/max(p,p′) ≤ 4Σ(√p−√p′)² ≤ 4·KL(p‖p′), this is ≤ (8e^bΔ²/μ²)·(L̂(π)+L̂(π°)). The noise is self-bounding.
3. Gaussian tails and a union bound: L̂(π̂) ≤ L̂(π°) + √(a·(L̂(π̂)+L̂(π°))) with a = 16e^b(Δ/μ)²ι, so L̂(π̂) ≤ 3L̂(π°) + a.
4. Bernstein with Var ≤ G·L(π): |L̂−L| ≤ L/2 + (4/3)Gι/T for all π. Combine. ∎

**Remarks.**
- The privacy term is independent of T: per-release noise grows as √T and averaging removes exactly that. This is P2 (square-root cancellation) as a rate statement. With Δ = 2Λ/M the term is 128·e^b·Λ²·ι/(Mμ)²: consistency comes from shards, not queries.
- Privacy-dominated threshold for weighted full release: T* ≍ G·μ²/(e^b·Δ²) ≍ (Mμ)²/(e^b·Λ). Queries beyond T* buy nothing. This threshold does not apply to sampled feedback (Proposition 6′).
- H = 1 with public contexts is easier than the paper's setting: C_app is an absolute constant and no clipping is needed. The paper's B factor in C_app comes from policy-dependent state distributions (H ≥ 2).

**Proposition 6′ (sampled feedback; replaces the first-pass Corollary 6).** Same setting, but at each context draw a_t ~ q_{s_t} and release ℓ*_{s_t}(a_t) + N(0, Δ²T/μ²). Use the importance-weighted estimator centred at c = log q. Let Λ_c = sup |ℓ*_s(a) − log q_s(a)| over supp q, and

  V̄ = sup_{π,π°} E_s[Σ_a (π−π°)²(ℓ*−log q)²/q] / (L(π)+L(π°)) ≤ 8·e^b·Λ_c².

With probability ≥ 1 − 3δ,

  L(π̂) ≤ 3·L* + [4G + 4V̄ + (8/3)(b + e^b·Λ_c) + 2e^b]·ι/T + 65·e^b·(Δ/μ)²·ι.

Proof sketch. Bernstein for the sampling part with variance ≤ (G + V̄)(L(π)+L(π°)) and range 2(b + e^bΛ_c). The privacy part is Gaussian given the actions with variance (Δ/μ)²·(1/T)Σ_t (π−π°)²(a_t)/q²(a_t); that empirical average is ≤ 2·8e^b(L(π)+L(π°)) + e^{2b}ι/T by Bernstein for a bounded non-negative variable. Combine as in Theorem 5 step 3. Constants are not optimized. ∎

What this says:
- The privacy term has the same order as for weighted full release (65e^b against 32e^b). That part of the first-pass claim survives.
- The T-term is governed by V̄ and e^b·Λ_c, which measure log-ratios at rarely sampled actions, not by G. Sampled feedback matches weighted full release only for T ≫ (V̄ + e^bΛ_c)·μ²/(e^bΔ²), and that can be arbitrarily larger than T*.
- A homoscedastic full release replaces e^b by up to d in the privacy term and can then be worse than sampling. Reversal is an artefact of uniform noise.

**Counterexample to the first-pass Corollary 6 (from the review, reproduced in check R1).** Two actions, teacher p* = (e^{−R}, 1 − e^{−R}), candidates A = q = (1/R, 1 − 1/R) and B = (0, 1), sampling law q. Then G ≈ 1 and e^b ≈ 1, so T* ≈ 2500 at Δ/μ = 0.02 for every R, but V̄ ≈ R. With R = 10⁶ and T = 10⁵ the sampled estimator has risk 0.905 (it picks A whenever the rare action is never drawn, probability 0.905) while weighted full release has risk 0. Across R ∈ {10², 10⁴, 10⁶} the sampled risk is about 0.90, 0.37 and 0 at T = R/10, R and 10R.

**Proposition 8 (lower bounds; proper learners, μ ≤ 1, richness R_k for the named pairs).**
- Realizable: d even, s ∈ {±1}^d balanced, Π = {p₊, p₋} with p_± ∝ exp(±ρ·s). Then ‖log p₊ − log p₋‖_∞ = 2ρ and KL(p₊‖p₋) = 2ρ·tanh ρ. Take 2ρ = kΔ with k = ⌈1/μ⌉: the two datasets are ≤ 2-GDP apart, so for any μ-GDP learner and any T the worst-case risk is ≥ c·min{1, (Δ/μ)²}.
- C_app = 1: candidates π, π′ and teachers p_± ∝ p₀·exp(±η·sign(π′−π)) with 4η = kΔ. Then KL(π‖p_±) − KL(π′‖p_±) = c₀ ± η‖π−π′‖₁, and c₀ = 0 for a mirror-symmetric pair with p₀ ∝ √(ππ′). Excess risk ≥ c·min{1, (Δ/μ)·‖π−π′‖₁}.

Both families consist of normalized distributions. The bounds are over these two-point families and for learners that output an element of Π.

So the floor is quadratic in Δ/μ when a constant C_app > 1 is allowed (Theorem 5 matches up to e^b·ι) and linear when C_app = 1.

## 4. Gradient version (prediction for qbopd)

On-policy RKL gradient in logit space at one position: the privacy-noise part is Σ_a p_a ξ_a (e_a − p). Write x_a = ‖e_a − p‖.

**Proposition 7.** Trace of the noise covariance, in units of Δ²/β:
- sampled token: E_p[x²] = 1 − ‖p‖² (this is P1);
- design w: Σ_a p_a²x_a²/w_a; optimum w ∝ p_a·x_a gives (E_p[x])²;
- homoscedastic full: d·Σ_a p_a²x_a².

Sampled/optimal = 1 + CV²(x under p) ≥ 1.

| student (d = 1000) | sampled | optimal | homoscedastic full | sampled/optimal |
|---|---|---|---|---|
| uniform | 0.999 | 0.999 | 0.999 | 1.00 |
| Zipf(1) | 0.971 | 0.968 | 24.5 | 1.00 |
| p_max = 0.9 | 0.190 | 0.050 | 8.13 | 3.8 |
| p_max = 0.99 | 0.0199 | 0.0006 | 0.098 | 34.6 |

The gain exists only at confident positions, where half the precision goes to the top token and the rest in proportion to p. It is also free of sampling variance. Zero-GPU gate: compute E_p[x²]/(E_p[x])² on logged student distributions from existing qbopd rollouts; if the position-averaged ratio is below ~1.5, drop the idea.

Caveat: the heavily noised low-probability coordinates cannot be clipped after release without bias (the P1 failure boundary applies).

## 5. Corrections to earlier statements (this thread)

- "Peaked students → reversal, best k ≈ 1/‖p‖₂²": refined. Optimally weighted full release never loses to sampling; the right object is precision ∝ p, not a top-k cutoff.
- "Floor ≈ HΔ/μ": that is the C_app = 1 floor. With constant C_app the floor is quadratic.
- §6's heuristic floor "linear in H" for the reverse-KL plug-in: wrong. The plug-in pays H² (Theorem 9 and the instance in §9); linear in H needs a different estimator (Theorem 10).
- Conjecture ④ ("the sampled-vs-full gap may collapse to the privacy rate"): the privacy terms agree in order under cube calibration, but the sampled estimator's T-term can stay dominant far beyond T* (Proposition 6′). Under ℓ2-ball calibration full release keeps an advantage (Proposition 4).
- First-pass Corollary 6 ("past T* the feedback model is irrelevant"): wrong as stated; see Proposition 6′ and the counterexample.
- "Minimax-optimal over all GDP mechanisms" (first-pass L2) and "even the best possible mechanism" (first-pass §9): withdrawn. The supported claims are optimality within diagonal cube-calibrated designs, and worst-case lower bounds for zero-sum functionals.

## 6. H > 1 (first-pass heuristic, superseded by §9)

The first pass guessed a floor of order e^b·H·(Δ/μ)²·ι for the reverse-KL plug-in and listed clipping of noisy trajectory losses as the obstacle. Both points changed: with public dynamics no clipping is needed, and the plug-in's floor is H², not H. See §9.

## 7. Open problems, in order of value

1. **Main line (partly resolved in §11–12).** The shared-mass tournament gives C·L* with a floor linear in H when the pathwise KL bound B̄ is O(1); the coverage issue is settled by the design quantity Γ_T (§12). Still open: (a) the B̄ factor is real for the tournament (trap instance, §12) and the proof-calibrated hybrid only removes it when noise is near zero, so an adaptive rule that keeps a constant approximation factor without B̄ and a floor linear in H is the main open question, together with a lower bound saying whether that is possible; (b) usable constants; (c) a sampled implementation with clipped prefix weights and its rollout complexity; (d) infinite classes and a gradient-based analogue.
2. Normalization-aware mechanisms. The cube ignores Σ_a p*_a e^{δ_a} = 1; the true width of a single policy's cross-entropy is Δ·min_λ‖π − λp*‖₁. This also covers policy-level release, where scores computed inside the trust boundary have sensitivity about Δ‖π−π°‖₁. Is there a self-bounding private selection with floor H(Δ/μ)²ι and no e^b?
3. A log|Π| factor in the Proposition 8 lower bound (packing), and the extension to improper learners.
4. Optimal non-diagonal mechanisms for vector targets (gradient): a factorization problem with ‖·‖_{∞→2} sensitivity.
5. Subsampled shards: the model above uses worst-case composition (A5); redo the floor with the PLD accountant.

## 8. Numerical checks (12 Nov 2026 run)

- L1: deterministic variance 0.2843 vs theory 0.2844; sampled 21.00 vs 21.03.
- L2: optimum 1.000 = ‖c‖₁²; uniform 3.17; best of 20,000 random designs 2.07.
- P3: deterministic class Γ = 24.00 = d; three random classes all inside [1, min{d, |C|, e^b}].
- Inequality in T5 step 2: worst ratio 0.40 ≤ 1 over 200,000 random pairs.
- P7: Monte Carlo 0.00017 vs 0.00017 (optimal), 0.01101 vs 0.01081 (sampled).
- T5/P6′ (d = 64, |Π| = 128, e^b = 3.95, realizable, one context; a benign instance where the teacher is close to the design, so V̄ is small), mean risk:

| Δ/μ | weighted full (any T) | homoscedastic full | sampled T=10 | T=10² | T=10³ | T=10⁵ |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.00023 | 0 | 0 | 0 |
| 0.005 | 0.00004 | 0.00024 | 0.00028 | 0.00008 | 0.00004 | 0.00004 |
| 0.02 | 0.00091 | 0.00259 | 0.00094 | 0.00091 | 0.00087 | 0.00086 |
| 0.05 | 0.00504 | 0.01548 | 0.00447 | 0.00506 | 0.00514 | 0.00502 |
| 0.1 | 0.02016 | 0.06222 | 0.01754 | 0.02003 | 0.02016 | 0.02115 |
| 0.2 | 0.07772 | 0.18940 | 0.07143 | 0.07604 | 0.08105 | 0.07901 |

  Log-log slope of the floor in Δ/μ: 1.94 (theory 2). The floor is flat in T; sampled feedback differs from weighted full only when Δ/μ ≤ 0.005 and T ≤ 100; homoscedastic full is about 3× worse. The Theorem 5 bound is loose by roughly 150× in constants on this instance.

## 9. H > 1 (second pass, 12 Nov 2026)

**Model K (public dynamics).** Finite tabular or tree MDP whose dynamics and initial law are public (LLM case: a finite public prompt set, state = prefix). Write z = (h, s, a) and ν^π(z) = d^π_h(s)·π_h(a|s), a public function of π with total mass H. Then

  L(π) = Σ_z ν^π(z)·(log π(z) − ℓ*(z)),

linear in the private ℓ*. A design is a probability w over z; the release is y(z) = ℓ*(z) + N(0, Δ²/(μ²·w(z))), total cost μ-GDP by the §1 argument. A sampled implementation draws z_1..z_N ~ w by rolling out a *public* reference policy (a candidate, a mixture of candidates, a public base model); by Lemma 1 the privacy noise is unchanged and the extra error is O(1/N). Rollouts cost computation, not privacy, so **in Model K there is no statistical T-term: the privacy floor is the whole problem.** This also removes the clipping obstacle: heavy-tailed importance weights only affect how many rollouts are needed. Sampling states from the private teacher is different: it is itself a release and needs its own budget, which this accounting does not cover.

**Theorem 9 (plug-in reverse-KL ERM).** π̂ = argmin_π Σ_z ν^π(z)(log π(z) − y(z)). If H·w ≥ e^{−b}·max(ν^π, ν^{π*}) for every π ∈ Π, then with probability ≥ 1 − δ, ι = log(|Π|/δ):

  L(π̂) ≤ 3·L* + 16·e^b·H²·(Δ/μ)²·ι.

Proof: steps 2–3 of Theorem 5 with Σ_z(√ν^π − √ν^{π*})² = Σ_h Σ_{s,a}(√ν^π_h − √ν^{π*}_h)² ≤ Σ_h KL(ν^π_h‖ν^{π*}_h) ≤ H·L(π), by data processing at each step. ∎

The extra H is real for this estimator. A deviation at step 1 shifts the occupancy of every later state, and the linear plug-in pays for teacher noise at all of them. On the product-tilt pair (tilt ±ρ at every state), every diagonal cube-calibrated design gives the plug-in difference noise ≥ Δ‖ν^π − ν^{π′}‖₁/μ ≈ Δ·ρ·H^{1.5}/μ against a gap ≈ Hρ²/2, so the plug-in needs ρ ≳ (Δ/μ)√H and has risk ≈ H²(Δ/μ)².

This is not a lower bound for private mechanisms. Theorem 10 beats it in the same channel. A shared-parameter product tilt can be identified from the root-node log-odds alone. And by the normalization remark after Lemma 2, the occupancy-shift part of the functional, (d^π − d^{π′})(s)·⟨π_s, ℓ*_s⟩, has little genuine sensitivity when π_s is close to the teacher, so a normalization-aware mechanism need not pay for it. The distinction is between choosing an estimator inside a fixed channel and optimizing the whole mechanism.

**Theorem 10 (log-space least squares).** π̂ = argmin_π Σ_z w(z)(y(z) − log π(z))². This is the maximum-likelihood estimator for the Gaussian release channel. Let D_w(π, π′)² = Σ_z w(z)(log π(z) − log π′(z))². With probability ≥ 1 − δ:

  D_w(π̂, π*)² ≤ 2·min_π D_w(π, π*)² + 16·(Δ/μ)²·ι   (8·(Δ/μ)²·ι if π* ∈ Π),

with no dependence on H, d or coverage. If moreover d^π̂_h(s)·max(π̂, π*)(a|s) ≤ κ·H·w(z) for all z, then L(π̂) ≤ κ·H·D_w(π̂, π*)². In the realizable case:

  L(π̂) ≤ 8·κ·H·(Δ/μ)²·ι.

Proof. (i) Finite-class least squares in a Gaussian sequence model with noise σ²/w(z), σ = Δ/μ: ⟨ξ, θ − θ°⟩_w ~ N(0, σ²‖θ − θ°‖_w²), union bound, then x² ≤ a² + 2s(x + a) with s = σ√(2ι) gives x ≤ a + 2s. (ii) Chain rule L(π) = Σ_h E_{d^π_h}[KL(π_h‖π*_h)] and KL(p‖q) ≤ KL(p‖q) + KL(q‖p) = Σ_a |p_a − q_a|·|log(p_a/q_a)| ≤ Σ_a max(p_a, q_a)·log²(p_a/q_a). ∎

**Proposition 11 (lower bounds; proper learners, normalized teachers, richness as in Proposition 8).** Realizable: use the pair p_± of Proposition 8 at every state, with 2ρ = kΔ; then KL = H·2ρ·tanh ρ between two teachers that are ≤ 2-GDP apart, so risk ≥ c·min{1, H(Δ/μ)²}. C_app = 1: two candidates with disjoint occupancies and the per-state construction of Proposition 8 give excess risk ≥ c·min{·, HΔ/μ}.

**Summary.**

| Estimator | Approximation term | Privacy floor | Coverage needed |
|---|---|---|---|
| Plug-in reverse-KL ERM (T9) | 3·L* | 16·e^b·H²·(Δ/μ)²·ι | design covers occupancies of candidates and teacher |
| Log-space least squares (T10) | 2·κ·H·min_π D_w(π, π*)² | 16·κ·H·(Δ/μ)²·ι | design covers the selected student's states, and teacher/student action laws per step |
| Lower bound (P11) | — | H·(Δ/μ)² realizable; H·Δ/μ if C_app = 1 | — |

In the realizable case the floor is Θ̃(H(Δ/μ)²) and least squares attains it up to κ·ι. Its weakness is the approximation term: log-space L2 penalizes a mode-seeking student that ignores teacher tails, where L* would be small. Reverse KL is used in practice for exactly that case, so Theorem 10 is weak under misspecification.

**Consequences.**
- *Online vs offline.* With a public reference policy, the rollout source enters only through κ, the coverage of the selected student's state occupancy; on-policy or mixture-of-candidates rollouts keep κ small near the teacher. Rollouts sampled from the private teacher are not free: they are a release with its own budget, on top of needing the teacher to cover the student (the paper's Assumption 2.2). Measured κ (check R4, design d_mix ⊗ q̄): 1.0–1.2 for the teacher and its nearest candidates at every H, but the maximum over the class grows from 1.6 at H = 2 to 18.9 at H = 12. The linear-in-H floor is therefore a local statement: it holds when the selected policy stays in the well-covered part of the class.
- *Practical hypothesis.* With DP-noised teacher log-probs, precision-weighted squared-error matching of log-probs is the likelihood-correct objective and should beat the reverse-KL plug-in more as rollouts get longer. Caveats: this is finite-class selection, not gradient training; it assumes normalized log-probs; and the weights depend on the noise level. Test on the analytic categorical simulator before any GPU run.

**Numerical checks (exact Gaussian computations on a binary tree MDP, σ = Δ/μ = 0.01).**

- Inequality in T10 (ii): worst ratio 0.40 ≤ 1 over 200,000 random pairs.
- Two-policy product tilt, worst-case expected risk over the tilt size, in units of σ²:

| H | plug-in ERM | plug-in, best design | log-space LS | κ |
|---|---|---|---|---|
| 2 | 0.94 | 0.67 | 0.66 | 1.42 |
| 6 | 7.02 | 4.41 | 1.99 | 2.21 |
| 10 | 18.29 | 11.17 | 3.31 | 2.55 |
| 14 | 34.55 | 20.39 | 4.64 | 2.78 |

  Log-log slopes for H ≥ 6: 1.89, 1.81 and 1.00 (theory 2, 2, 1). Monte Carlo at H = 6 matches the error-probability formulas (0.2206 vs 0.2209; 0.0754 vs 0.0754).
- 64-policy class (teacher plus 63 tilts), mean risk in units of σ²:

| H | plug-in ERM | log-space LS | ratio |
|---|---|---|---|
| 2 | 2.86 | 2.04 | 1.4 |
| 6 | 29.86 | 9.56 | 3.1 |
| 12 | 130.90 | 19.81 | 6.6 |

  Slopes for H ≥ 4: 2.10 and 1.08.

## 10. External review, 13 Nov 2026: verdicts and changes

| Review point | Verdict after checking | Change |
|---|---|---|
| ① Corollary 6: equal privacy variance does not give equal risk; rare-action counterexample | Correct. Reproduced: sampled risk 0.905 at T = 10⁵ ≫ T* ≈ 2500, weighted full 0 | Corollary 6 replaced by Proposition 6′ with the second-moment parameter V̄; counterexample added to §3 |
| ② Cube sensitivity is not the richness needed for lower bounds; normalized log-probs cannot all move one way | Correct. The Gaussian mechanism calibrated to the cube stays a valid DP upper bound, but the first-pass L2/P8 shifts were not realizable | L2 lower bound now only for zero-sum c; P4 restated as a comparison of calibrated mechanisms; P8/P11 rebuilt from normalized two-point families |
| ③ H² is evidence about the plug-in in this channel, not about all mechanisms | Correct | §9 wording restricted; "best possible mechanism" withdrawn |
| ④ Experiments cover only realizable, finite, public-design cases | Correct | Misspecification experiment and κ measurement added below; teacher rollouts flagged as a release |
| Minor: small discrepancies in the §8 selection table | Expected | That table is Monte Carlo (4,000 trials) with no standard errors, and NumPy's Generator does not guarantee identical streams across versions. check_review.py reports standard errors |

**Scope of the scripts.** They verify risk inside the release channel (given Gaussian noise of the stated variance). They do not verify the privacy guarantee of a shard-ensemble teacher across neighbouring datasets; that rests on the sensitivity assumption S∞ and the accounting in claude/dp-opd-ier.md.

**Misspecification experiment (check R3).** Ternary tree MDP. Teacher: two-action head plus a tail action of mass 0.5–1.5% at every state; not in the class. Public class: 32 tail-ignoring candidates (tail mass 10⁻⁶, small head shift) and 32 head-distorting candidates (exact tail, larger head shift), built so that reverse KL prefers a tail-ignoring candidate and log-space L2 prefers a head-distorting one. Mean risk relative to L*, 1,000 trials, standard errors ≤ 0.12 in these units:

| H | σ = Δ/μ | plug-in / L* | least squares / L* |
|---|---|---|---|
| 2 | 0.01 | 1.01 | 3.96 |
| 2 | 0.1 | 3.01 | 4.43 |
| 2 | 0.3 | 7.42 | 6.68 |
| 4 | 0.03 | 1.42 | 4.07 |
| 4 | 0.1 | 5.71 | 5.08 |
| 8 | 0.01 | 1.06 | 4.05 |
| 8 | 0.03 | 2.79 | 4.35 |
| 8 | 0.1 | 8.50 | 5.58 |
| 8 | 0.3 | 10.14 | 7.86 |

- Least squares never recovers L*: it sits at about 4·L* with no noise at all, and almost never selects a tail-ignoring candidate (0% of trials up to σ = 0.1). The factor 4 is a property of the construction and could be made larger.
- The plug-in is at L* for small noise and degrades faster. The crossover noise level falls with H: about 0.3 at H = 2, between 0.03 and 0.1 at H = 4 to 8.
- Neither estimator dominates. Least squares wins only once privacy noise dominates the approximation gap. Risks saturate at large σ because the class is finite.
- κ in this class: median 3.2 to 13.9 and maximum 4.0 to 63.4 from H = 2 to 8.

**Normalization check (R2).** Feasible first-order width of ⟨c, log p*⟩ relative to the cube value: 0.76 for a random policy, 0.04 for a policy 5% from the teacher, 0.94 on average (minimum 0.69) for differences of two policies.

**What the note now supports.** The design identities and optimal precision allocation within diagonal Gaussian designs; the gradient-noise allocation; the H = 1 upper bound with a T-independent privacy term for weighted full release; the corrected sampled-feedback bound; the two multi-step upper bounds and their different horizon dependence in this channel; two-point lower bounds for proper learners. It does not support "privacy eliminates the difference between feedback modes" or any H² lower bound.

**Next.** Open problem 1 in §7 is the main line. No GPU runs before it and the lower-bound extensions are settled.

## 11. Shared-mass tournament (third pass, 13 Nov 2026)

Question from §7 and the review: in the same Gaussian release channel, can a selector keep reverse KL's approximation term while paying H instead of H²? Yes, under a bounded pathwise KL sum. The proof below is my own and has not been externally checked.

**Setting.** Model K with a tree MDP (state = prefix, the LLM case), teacher with full support, finite Π, learner outputs an element of Π. Write k_h(s; π) = KL(π_s‖π*_s) and B̄ = sup_{π∈Π} max over trajectories of Σ_h k_h(s_h; π).

**Definitions.**
- Shared-mass truncated risk: T_{π′}(π) = Σ_s min(d^π(s), d^{π′}(s))·k(s; π) = E_{P^π}[Σ_h min{1, R_h}·k_h(s_h; π)], with R_h = d^{π′}(s_h)/d^π(s_h). Always T_{π′}(π) ≤ L(π). The weight min{1, R_h} is the clipped prefix importance weight of the paper's Algorithm 1; as far as my extraction shows, T is the same kind of object as its Trunc operator.
- Pairwise statistic: Δ(π, π′) = T_{π′}(π) − T_π(π′) = Σ_s min(d^π, d^{π′})(s)·[k(s; π) − k(s; π′)]. It compares the two candidates at the same states, on the occupancy they share, and is linear in ℓ*, so Δ̂ is obtained by replacing ℓ* with y.
- Tournament: π̂ = argmin_π max_{π′} Δ̂(π, π′).

**Lemma 12 (transfer).** For any π, π′ ∈ Π: L(π) ≤ (2 + 79·B̄)·T_{π′}(π) + 56·B̄·L(π′).

Proof. Let τ be the first h with R_h < 1/2 (τ = ∞ if none). For h < τ the weight min{1, R_h} ≥ 1/2, so E_π[Σ_{h<τ} k_h] ≤ 2T. On {τ ≤ H} the remaining sum is ≤ B̄. Hence L(π) ≤ 2T + B̄·P_π(τ ≤ H).

Write R_h = V_h·W_h with W_h = d*(s_h)/d^π(s_h) and V_h = d^{π′}(s_h)/d*(s_h). At time τ, either W_τ < 1/√2 or V_τ < 1/√2. Let φ(w) = (1 − √w)² and c₀ = φ(1/√2) = 0.0253.
- Case W_τ < 1/√2. With σ = τ∧H, Markov gives P ≤ E_π[φ(W_σ)]/c₀. Pointwise φ(w) ≤ −log w + (w − 1), and E_π[W_σ] ≤ 1 (W is a non-negative supermartingale under P^π), so E_π[φ(W_σ)] ≤ E_π[−log W_σ] = E_π[Σ_{j<σ} k_j(s_j; π)] ≤ E_π[Σ_{j<τ} k_j] ≤ 2T. So this case has probability ≤ 79·T.
- Case V_τ < 1/√2 ≤ W_τ. First-crossing states form an antichain, and on them d^π(s) ≤ √2·d*(s). So the probability is ≤ √2·P_*(∃h: V_h < 1/√2). Under P*, V is a martingale and φ(V_h) a non-negative submartingale; Doob's inequality gives P_*(·) ≤ E_*[φ(V_H)]/c₀ ≤ KL(P^{π′}‖P*)/c₀ = L(π′)/c₀. So this case has probability ≤ 56·L(π′). ∎

The point of the stopping-time argument is that π's cost for leaving the shared region is paid at shared states, so it is charged to T, not to L(π).

**Lemma 13 (self-bounding noise).** If H·κ·w(s, a) ≥ min(d^π, d^{π′})(s)·max(π, π′)(a|s) for all pairs and all z, then Var[Δ̂(π, π′)] ≤ 8·κ·H·(Δ/μ)²·(T_{π′}(π) + T_π(π′)).

Proof. Var = σ²·Σ_{s,a} m(s)²(π−π′)²(a|s)/w(s,a) ≤ σ²κH·Σ_s m(s)·Σ_a (π−π′)²/max(π,π′), and Σ_a (π−π′)²/max(π,π′) ≤ 4Σ_a(√π − √π′)² ≤ 8Σ_a[(√π − √π*)² + (√π′ − √π*)²] ≤ 8[k(s;π) + k(s;π′)]. ∎

Here κ involves the candidates and the design only. It is public and can be computed before any release; no coverage of the teacher is assumed.

**Theorem 14 (tournament).** Under the coverage condition of Lemma 13, with probability ≥ 1 − δ and ι = log(|Π|²/δ):

  L(π̂) ≤ (14 + 609·B̄)·L* + (64 + 2528·B̄)·κ·H·(Δ/μ)²·ι.

Proof. Let a = 16κH(Δ/μ)²ι; on the good event |Δ̂ − Δ| ≤ √(a·(T_{π′}(π) + T_π(π′))) for all pairs. Let π^b attain L*. For any π′, with u = T_{π′}(π^b) ≤ L* and t = T_{π^b}(π′): Δ̂(π^b, π′) ≤ u − t + √(au) + √(at) ≤ 2L* + a/2. So Δ̂(π̂, π^b) ≤ max_{π′} Δ̂(π̂, π′) ≤ 2L* + a/2. With x = T_{π^b}(π̂) and T_{π̂}(π^b) ≤ L*: x ≤ 3L* + a/2 + √(a(x + L*)), hence x ≤ 7L* + 2a. Apply Lemma 12 with π′ = π^b. ∎

**Reading.**
- The approximation term is in reverse KL, with C_app = O(1 + B̄). The paper's non-private C_app is Θ(B) for its second-moment parameter B, so a factor of this kind is not surprising, but I have not shown B̄ is necessary here.
- The floor is O((1 + B̄)·κ·H·(Δ/μ)²·ι). With B̄ and κ bounded it is linear in H, matching the realizable lower bound of Proposition 11 up to κ·ι. If B̄ grows like H (constant local KL on every path) the floor is of order H² again, and Theorem 9 is then the better bound.
- Why it works: the plug-in compares full occupancies, and its H² comes from the occupancy-shift term of §7. The tournament never estimates that term. Lemma 12 shows it is controlled by the shared-mass risk and the other candidate's risk.
- Cost: |Π|² pairwise statistics from the same release, so no extra privacy, only computation.
- Coverage: κ·H is a loose stand-in for the public design quantity Γ_T(w) of §12, which stays close to H where κ·H grows quickly.
- Limits: finite class, proper learner, exact design with public dynamics, crude constants (the intermediate bound in the proof is within a factor 1.3 of tight on the random classes tested, the final constants are far from it).

**Numerical checks (check_tournament.py).**
- Lemma 12 on 12 random classes (binary tree, H = 7, all ordered pairs): max LHS/RHS = 0.40; the intermediate bound L ≤ 2T + B̄·P(crossing) has max LHS/RHS = 0.79. Lemma 13: max ratio 0.29. All ≤ 1 as required.
- Realizable 64-policy class, σ = 0.01, mean risk in units of σ² (standard error):

| H | plug-in | least squares | tournament | κ | B̄ |
|---|---|---|---|---|---|
| 2 | 2.93 (0.10) | 2.15 (0.08) | 2.16 (0.08) | 1.7 | 0.08 |
| 6 | 30.58 (1.16) | 9.11 (0.33) | 9.02 (0.32) | 2.7 | 0.24 |
| 12 | 134.15 (5.41) | 20.66 (0.80) | 20.43 (0.78) | 7.5 | 0.52 |

  Slopes in H for H ≥ 4: 2.04, 1.14, 1.14.
- Misspecified class (the §10 construction, fresh draw), mean risk / L*:

| H | σ | plug-in | least squares | tournament |
|---|---|---|---|---|
| 2 | 0.03 | 1.12 | 4.22 | 1.08 |
| 4 | 0.03 | 1.45 | 4.02 | 1.12 |
| 8 | 0.01 | 1.06 | 4.08 | 1.01 |
| 8 | 0.03 | 2.81 | 4.30 | 1.14 |
| 8 | 0.1 | 8.57 | 5.57 | 3.99 |
| 8 | 0.3 | 9.75 | 7.99 | 7.50 |

  Standard errors ≤ 0.15. The tournament is at L* with no noise, like the plug-in, and its mean risk was no higher than either alternative in all 20 cells of the full table (at σ = 0.3 the gap to least squares is within about one standard error for H ≤ 6). This is one constructed family; κ here grows from 4 to 36 and B̄ from 0.3 to 1.3 between H = 2 and 8.

**A remark on practice (heuristic, not analysed).** The exact gradient of the trajectory-level reverse KL weights each step's score by the log-ratio cost-to-go over all later steps, whereas on-policy distillation as usually implemented uses only the local term at visited prefixes. The local term is the gradient-level counterpart of a same-state comparison; the cost-to-go term is the counterpart of the occupancy shift. Under noised teacher log-probs this suggests the local objective is the one whose noise scales linearly with rollout length.

## 12. Fourth pass, 15 Nov 2026: proof check, the B̄ factor, a hybrid, and the design

**12.1 Step-by-step check of Lemma 12.** check_hybrid.py evaluates every inequality in the proof exactly (no sampling) on 800 ordered pairs from four families: both policies near the teacher, unrelated near-deterministic policies, a teacher with one branch nearly removed, and a policy that differs from the teacher only deep in the tree. Worst ratio of left side to right side per step:

| Step | Worst ratio |
|---|---|
| L ≤ E[Σ_{h<τ} k_h] + B̄·P(τ ≤ H) | 1.0000 |
| E[Σ_{h<τ} k_h] ≤ 2T | 0.904 |
| c₀·P(case 1) ≤ E φ(W_σ) | 0.389 |
| E φ(W_σ) ≤ E[−log W_σ] | 0.952 |
| E[−log W_σ] = E[Σ_{j<σ} k_j] (absolute difference) | 2.7·10⁻¹⁵ |
| P(case 2) ≤ √2·P_*(V falls below 1/√2) | 0.995 |
| c₀·P_*(V falls below 1/√2) ≤ E_* φ(V_H) | 0.639 |
| E_* φ(V_H) ≤ L(π′) | 0.945 |
| Final: L ≤ (2 + 79B̄)T + 56B̄·L(π′) | 0.343 |

All hold, and several are close to tight, so the families do exercise the steps. The first run of this check reported violations; the cause was a bug in the checker (one family built unnormalized policies), now fixed and guarded by an assertion. This is numerical evidence, not a substitute for an independent read of the proof.

**12.2 The B̄ factor is real for the tournament.** Trap instance (binary tree): the teacher takes one root action with probability 5%. Good candidates almost never take it (probability 0.1%) and otherwise follow the teacher. Trap candidates match the root but behave badly inside that subtree. The two families share almost no mass in the subtree, so the pairwise statistic cannot see the trap candidates' loss there. With no noise at all:

| H | B̄ | plug-in / L* | tournament / L* |
|---|---|---|---|
| 4 | 1.7 | 1.00 | 1.30 |
| 8 | 3.8 | 1.00 | 3.67 |

So C_app growing with B̄ is a property of the tournament, not of the proof. This is the same phenomenon as the paper's failure result for objectives evaluated on other policies' states.

**12.3 Proposition 15 (hybrid).** Let s(π, π′) = (Δ/μ)·√(Σ_z (ν^π − ν^{π′})²/w(z)); it is a public number. With z = √(2ι), keep the survivors S = {π : L̂(π) − L̂(π′) ≤ z·s(π, π′) for all π′} and run the tournament inside S. With probability ≥ 1 − 2δ:
- the best candidate π^b survives, so Theorem 14's bound holds for π̂;
- L(π̂) ≤ L* + 2z·s(π̂, π^b), and if H·w ≥ e^{−b}·max(ν^π, ν^{π′}) for all pairs, L(π̂) ≤ 3L* + 64·e^b·H²·(Δ/μ)²·ι.

Proof. On the plug-in noise event, L̂(π^b) − L̂(π′) ≤ L* − L(π′) + z·s ≤ z·s, so π^b ∈ S; Theorem 14's proof only needs π^b among the opponents. A survivor satisfies L(π̂) − L* ≤ 2z·s(π̂, π^b). Finally Σ_z(ν−ν′)²/w ≤ 4He^b·Σ_z(√ν − √ν′)² ≤ 8e^bH²(L(π) + L(π′)) by the Hellinger triangle inequality and data processing per step, and x − L* ≤ √(a′(x + L*)) with a′ = 64e^bH²(Δ/μ)²ι gives x ≤ 3L* + a′. ∎

This also removes the teacher-coverage assumption from Theorem 9: only the candidates' occupancies need covering.

In simulation the guarantee is weaker than it sounds. Risk relative to L* on the trap instance, H = 8:

| σ | plug-in | tournament | hybrid | hybrid, z = 1 (outside the guarantee) |
|---|---|---|---|---|
| 0 | 1.00 | 3.67 | 1.00 | 1.00 |
| 0.01 | 1.39 | 3.38 | 3.38 | 2.52 |
| 0.03 | 3.06 | 3.31 | 3.31 | 3.27 |
| 0.1 | 3.63 | 2.97 | 2.97 | 3.21 |
| 0.3 | 3.75 | 3.14 | 3.14 | 3.41 |

With the proof-calibrated threshold almost nothing is eliminated once σ ≥ 0.01, so the hybrid equals the tournament. On the earlier classes it also equals the tournament (no loss). A smaller threshold helps on the trap at σ = 0.01 but hurts elsewhere (realizable class, H = 12: 32.6 against 19.5 in units of σ²). Summary of the three selectors on what has been tested: the plug-in is best when noise is small and unshared regions matter; the tournament is best when privacy noise dominates and B̄ is small; no rule tested gets both.

**12.4 Design: Γ_T replaces κ·H.** For a pair p = (π, π′) let c_p(z) = min(d^π, d^{π′})(s)·(π − π′)(a|s) and S̃_p = Σ_s min(d^π, d^{π′})(s)·Σ_a (π−π′)²/max(π, π′). Both are public, and S̃_p ≤ 8(T_{π′}(π) + T_π(π′)) by the chain in Lemma 13. Define

  Γ_T(w) = max_p [Σ_z c_p(z)²/w(z)] / S̃_p.

Then Var[Δ̂_p] ≤ 8·Γ_T(w)·(Δ/μ)²·(T_{π′}(π) + T_π(π′)), and Theorem 14 holds with κ·H replaced by Γ_T(w). Facts: Γ_T(w) ≤ κ(w)·H; min_w Γ_T(w) ≥ max_p ‖c_p‖₁²/S̃_p; for a single pair the optimum is ≤ 2·Σ_s min(d^π, d^{π′})(s) ≤ 2H (Cauchy–Schwarz twice). Minimizing Γ_T over w is a convex program, solved here by multiplicative weights on pairs with the best response w ∝ √(Σ_p λ_p c_p²/S̃_p).

| Class | H | κ·H (mixture design) | Γ_T (mixture) | Γ_T (optimal) | single-pair lower bound | Γ_T(optimal)/H |
|---|---|---|---|---|---|---|
| realizable | 2 | 3.2 | 3.0 | 2.3 | 1.9 | 1.12 |
| realizable | 6 | 21.0 | 7.8 | 6.7 | 5.6 | 1.11 |
| realizable | 12 | 77.0 | 14.8 | 12.5 | 11.4 | 1.04 |
| misspecified | 2 | 8.0 | 5.7 | 3.5 | 2.2 | 1.77 |
| misspecified | 4 | 46.9 | 12.8 | 7.1 | 3.8 | 1.78 |
| misspecified | 8 | 228.2 | 20.6 | 12.9 | 7.4 | 1.61 |

The growth of κ with H reported in §10–11 was an artefact of bounding by a worst-case ratio. The quantity that enters the variance stays between 1.0·H and 1.8·H under the optimal design in both classes, so in these classes the floor of Theorem 14 is linear in H up to B̄. I have not rerun the selection experiments with the optimal design, and I have no bound on how min_w Γ_T grows with |Π| in general.

**12.5 Where this leaves the direction.** Supported: in this release channel, with bounded pathwise KL, reverse-KL approximation and a floor linear in H are compatible (Theorem 14 with Γ_T), and the floor matches the realizable lower bound in H. Not supported: any claim without the B̄ condition. The open question in §7 item 1(a) is now the sharpest one: either an adaptive selector that avoids the trap at moderate noise, or a lower bound showing that constant-factor reverse-KL approximation with a floor linear in H is impossible without a bound like B̄.
