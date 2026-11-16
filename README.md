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
| `reviews/2026-11-12-integration-points.md` | An earlier list of candidate DP integration points that the note responds to |
| `reviews/2026-11-13-external-review.md` | External review of the first two passes; §10 of the note answers it |

## Running the checks

```
pip install -r requirements.txt
cd checks
python check_feedback_design.py
python check_horizon.py
python check_review.py
python check_tournament.py
python check_hybrid.py     # imports check_tournament.py from the same directory
```

Each script prints its results; the numbers quoted in `theory.md` come from these runs. Monte Carlo values
can differ slightly across NumPy versions.

## Scope

The scripts verify risk inside the release channel, given Gaussian noise of the stated variance. They do not
verify the privacy guarantee of any particular teacher; that rests on the sensitivity assumption stated in §1.
