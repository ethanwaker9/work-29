import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("correctness of all six blind rotations", ["tests/test_rotation.py"]),
    ("gate bootstrapping microbenchmark", ["experiments/e1_blindrot.py"]),
    ("window and fusion depth sweep", ["experiments/e2_tradeoff.py"]),
    ("minimal gadget dimensions", ["analysis/optimize.py"]),
    ("confidential transfer workload", ["experiments/e3_contract.py"]),
    ("figures", ["figures/make_figures.py"]),
    ("tables", ["figures/make_tables.py"]),
]


def main():
    for name, args in STEPS:
        print("=" * 72)
        print(name)
        print("=" * 72, flush=True)
        r = subprocess.run([sys.executable] + [os.path.join(BASE, a) for a in args],
                           cwd=BASE)
        if r.returncode != 0:
            print("step failed:", name)
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
