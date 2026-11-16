The most worthwhile part to retain is the comparison of estimators under the same Gaussian release channel: in the replicated 64-policy experiments, the risk of the plug-in estimator grows substantially faster with H than that of log-space LS. However, this only validates discrepancies on specific realizable instances, rather than establishing an $H^2$ lower bound that applies to all private algorithms.

Add

Download

My assessment: This is worth pursuing as a theoretically oriented research note with substantial substance; it is not yet ready for external circulation as a research report where all main conclusions are proven. The most valuable components are the feedback precision allocation and the temporal-domain differences between the two estimators under the same Gaussian release channel. The most critical revisions needed are the feedback "collapse" conclusion in C6 and the scope of applicability for several lower bounds.

I have read the three attachments, ran two scripts as-is, and additionally performed counterexample tests. No changes have been made to the attachments.

1. Well-supported components
Section | Review verdict
L1: Equal privacy noise variance for matched designs | Algebraically correct and consistent with simulations; only establishes variance properties
L2 optimal precision allocation, P3 upper bound on Γ | Holds within the specified class of diagonal Gaussian designs
P7: Optimal allocation of gradient noise | Derivations match simulations, representing a practically testable direction
T5: Single-step risk upper bound | Proof holds under public design, coverage, and bounded KL assumptions
T9, T10: Multi-step plug-in and LS upper bounds | Derivations hold under Model K and their respective coverage conditions

The numerical results provide valid verification and are not merely decorative. Reproduction of the table in check_horizon.py: the H slope for the 64-policy experiment is 2.10 versus 1.08; at H=12, the risk is 130.90 versus 19.81, in units of $(\Delta/\mu)^2$. The noise scale slope for the feedback script is also 1.94. There are minor numerical discrepancies in the selection risk table, whose root cause remains unidentified.

That said, simulations only characterize behavior on these instances and cannot substitute for proofs of unified upper bounds or minimax lower bounds.

2. Four mandatory revisions
① C6 is not merely "abbreviated proof"; the existing assumptions are insufficient to support its conclusion

L1 states:
Equal privacy noise variance $\nRightarrow$ identical tail bounds, misselection probabilities, or risk.

The sampling noise is Gaussian conditioned on the action sequence; unconditionally, it forms a Gaussian mixture with random variance. Additional sampling error further depends on log-ratios over rare actions.

I have constructed and numerically verified a counterexample:

The target teacher distribution is $p^*=(e^{-R},1-e^{-R})$, with candidates $q=(1/R,1-1/R)$ and $(0,1)$, and sampling distribution $q$.
Set $R=10^6$, $\Delta/\mu=0.02$, so $G\approx1$, $e_b\approx1$, and the reported $T^*\approx2500$.
Yet at $T=100000$, the expected excess risk of the centralized sampling estimator remains approximately 0.905, while the risk of the matched full-feedback estimator is close to zero.

The reason is that roughly 90.5% of sampling sequences never observe the rare action that governs risk. The threshold lacks tail or second-moment parameters. This does not rule out convergence for fixed benign instances as $T\to\infty$, but invalidates the current unified threshold that relies solely on $G,e_b,\Delta/\mu$.

② The "cube sensitivity upper bound" does not equal the richness required for the lower bound

Valid perturbations of normalized log-probabilities must satisfy:
$$\sum_a p_a e^{\delta_a}=1$$
It is therefore impossible for all coordinates to increase strictly while preserving normalization. The $\Delta\mathrm{sign}(c)$ shift required in L2 for strictly positive $c$ is not realizable under the current teacher model. The symmetric shift construction around a single point in P8 also needs rewriting.

This does not mean the scaling order of quadratic or linear privacy lower bounds is necessarily incorrect—hard instances can be reconstructed using normalized binary exponential families. However, one cannot directly conclude that "all GDP mechanisms are optimal" from $S_\infty$. P4 should also explicitly discuss worst-case MSE rather than the minimum variance over arbitrary mechanisms; a constant output has zero variance.

③ Evidence for $H^2$ must be restricted to the studied plug-in mechanism

The upper bound in T9 and the tree experiments are valuable; the issue lies with the wording "best possible mechanism" in §9.

The emergence of $H^2$ under linear plug-in and the specified Gaussian release scheme does not imply that all private mechanisms must incur $H^2$ cost. For example, a special case of product-tilt with shared parameters can identify the teacher via root-node log-odds without accumulating noise across all states. A clear distinction must be drawn between "selecting an estimator within a fixed channel" and "optimizing the entire privacy mechanism".

④ Experiments do not yet cover the parts that determine practical utility

Validation so far is limited to realizable settings, finite candidate classes, and public designs. It has not established that:

LS outperforms reverse-KL under model misspecification when the student ignores the teacher’s tails;
$\kappa$ remains controlled as H or the policy class grows, so the linear H scaling holds in practice;
private teacher rollouts are free—the original teacher sampling itself may leak data and cannot be treated directly as a source of public states.

Furthermore, the scripts verify risk within the release channel, not the privacy guarantees of sliced teachers across adjacent datasets.

3. Recommended next steps
First refine the theoretical boundaries; GPU runs are not recommended at this stage.

Revise C6: add log-ratio tail conditions, rederive finite-sample thresholds, and include the rare-action example above for stress testing.
Reconstruct lower bounds: use valid normalized teachers, clearly define adjacency relations, population distances, parameter ranges, and whether outputs are restricted to $\Pi$.
Run misspecification experiments: fix a public candidate class, shift the teacher outside this class, and compare approximation error and privacy error between reverse-KL and LS.
Narrow the main research line: prioritize studying how to retain the misspecification advantage of reverse-KL while mitigating temporal dependence within the same DP Gaussian feedback channel. This is more rigorous and scientifically meaningful than claiming that privacy eliminates feedback-mode differences.

Summary: The core content is substantive, but the strength of the conclusions exceeds the strength of the supporting evidence. Retain the design identities, gradient allocation results, and estimator comparisons, and fix the quantifier issues in C6 and the lower bounds. Only then will this note serve as a reliable foundation for the next round of research.