# dp-rkl

Theory notes and numerical checks on combining differential privacy with reverse-KL policy distillation,
building on Liu, Wei, Zimmert, "Reverse-KL Policy Distillation: Statistical Guarantees and Limits"
(arXiv 2610.30844). The setting is a private teacher whose log-probabilities are released with Gaussian noise.

Status: a working research note, not a finished paper. Several proofs have not been externally checked;
`theory.md` marks the status of every result in its first table and lists corrections in §5 and §10.

## Contents

| Path | What it is |
|---|---|
| `theory.md` | The full note: model, results, proofs, corrections, open problems |
| `checks/check_feedback_design.py` | Single-step results: design identity, optimal precision allocation, rate theorem (§2–4, §8) |
| `checks/check_horizon.py` | Multi-step: plug-in vs log-space least squares on a tree MDP (§9) |
| `checks/check_review.py` | Checks added after external review: rare-action counterexample, normalization, misspecification (§10) |
| `checks/check_tournament.py` | Shared-mass tournament: transfer lemma, variance bound, simulations (§11) |
| `checks/check_hybrid.py` | Step-by-step proof check, trap instance, hybrid selector, design quantity (§12) |
| `checks/check_round2.py` | Class-complexity example with dual certificate, anchor-normalized plug-in, paired design × selector experiment (§13) |
| `checks/check_round3.py` | Counterexamples from the PR #1 re-review, a negative test of the failure detection, and the H × anchor-quality sweep (§14) |
| `checks/_checks.py`, `checks/_pairs.py`, `checks/run_all.py` | Failure assertions and the scale-invariant pair structure shared by the scripts; a single entry point |
| `reviews/2026-11-12-integration-points.md` | An earlier list of candidate DP integration points that the note responds to |
| `reviews/2026-11-13-external-review.md` | External review of the first two passes; §10 of the note answers it |

## Running the checks

```
pip install -r requirements.txt      # pinned: numpy 2.5.3, scipy 1.18.1 (run on Python 3.13)
cd checks
python run_all.py                    # all seven scripts; add --strict to turn RuntimeWarnings into errors
python check_round2.py               # or any single script
```

Every script exits non-zero if an inequality is violated, a comparison is out of tolerance, or a value is
not finite. The full run takes about three minutes.

Each script prints its results; the numbers quoted in `theory.md` come from these runs. Monte Carlo values
can differ slightly across NumPy versions.

## Scope

The scripts verify risk inside the release channel, given Gaussian noise of the stated variance. They do not
verify the privacy guarantee of any particular teacher; that rests on the sensitivity assumption stated in §1.
