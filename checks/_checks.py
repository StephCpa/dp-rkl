"""Shared failure detection for the check scripts.

Every script records its pass/fail conditions through a Checker and calls finish(), which exits with a
non-zero status if any condition failed or any recorded value was not finite. Printing alone is not a check.
"""
import sys
import numpy as np


class Checker:
    def __init__(self, name):
        self.name = name
        self.failures = []
        self.count = 0

    def finite(self, label, value):
        self.count += 1
        if not np.all(np.isfinite(np.asarray(value, dtype=float))):
            self.failures.append(f"{label}: non-finite value")

    def le(self, label, value, bound, tol=1e-9):
        """value <= bound (up to tol)."""
        self.finite(label, value)
        self.count += 1
        if not (float(value) <= float(bound) + tol):
            self.failures.append(f"{label}: {float(value):.6g} > {float(bound):.6g}")

    def close(self, label, value, target, rel=0.05, abs_tol=0.0):
        self.finite(label, value)
        self.count += 1
        if not (abs(float(value) - float(target)) <= rel * abs(float(target)) + abs_tol):
            self.failures.append(f"{label}: {float(value):.6g} not within {rel:.0%} of {float(target):.6g}")

    def between(self, label, value, lo, hi):
        self.finite(label, value)
        self.count += 1
        if not (lo <= float(value) <= hi):
            self.failures.append(f"{label}: {float(value):.6g} outside [{lo}, {hi}]")

    def argmin(self, label, scores, axis=1):
        """argmin that refuses non-finite scores. np.argmin on NaN silently returns an index, so a selector
        fed NaN would otherwise report the (finite) true risk of an arbitrary candidate."""
        self.count += 1
        scores = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(scores)):
            self.failures.append(f"{label}: non-finite score before argmin")
        return np.argmin(scores, axis=axis)

    def finish(self):
        if self.failures:
            print(f"\n{self.name}: {len(self.failures)} of {self.count} checks FAILED")
            for f in self.failures:
                print("   FAIL  " + f)
            sys.exit(1)
        print(f"\n{self.name}: all {self.count} checks passed")
