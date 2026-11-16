"""Run every check script and report which passed. Exit status is non-zero if any script fails.

Usage:  python run_all.py            # assertions inside each script decide pass/fail
        python run_all.py --strict   # additionally turn every RuntimeWarning into an error

Each script records its conditions through _checks.Checker and exits non-zero on any violated inequality,
out-of-tolerance comparison or non-finite value. --strict is off by default because some BLAS builds raise
spurious floating-point warnings inside matrix products; on the pinned environment it passes.
"""
import subprocess
import sys
import time

SCRIPTS = ["check_feedback_design.py", "check_horizon.py", "check_review.py", "check_tournament.py",
           "check_hybrid.py", "check_round2.py", "check_round3.py"]


def main():
    strict = "--strict" in sys.argv
    failed = []
    for name in SCRIPTS:
        cmd = [sys.executable] + (["-W", "error::RuntimeWarning"] if strict else []) + [name]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "(no output)"
        status = "ok  " if proc.returncode == 0 else "FAIL"
        print(f"{status} {name:28s} {time.time() - t0:6.1f}s   {last}")
        if proc.returncode != 0:
            failed.append(name)
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
    print(f"\n{len(SCRIPTS) - len(failed)} of {len(SCRIPTS)} scripts passed" + (" (strict)" if strict else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
