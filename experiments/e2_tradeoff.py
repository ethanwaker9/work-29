import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fhe.walk import build_stops, hop_cost

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def expected_units(n, N, w, m, trials=120, seed=17):
    rng = np.random.default_rng(seed)
    two_n = 2 * N
    tot = []
    kap = []
    for _ in range(trials):
        a = rng.integers(0, 1 << 18, n)
        x = -a.astype(np.float64) * two_n / float(1 << 18)
        y = 2 * np.rint((x - 1) / 2.0) + 1
        y = np.where(np.abs(x) < 1.0, 0.0, y).astype(np.int64) % two_n
        act = [i for i in range(n) if y[i] != 0]
        buckets, L = build_stops(y[act], two_n)
        occ = [s for s in range(2 * L) if buckets[s]]
        c = 1 + len(act)
        prev = None
        for s in occ:
            if prev is not None:
                if prev <= L - 1 < s:
                    c += hop_cost(L - 1 - prev, w, 0) + 1 + hop_cost(s - L, w, m)
                else:
                    c += hop_cost(s - prev, w, m)
            prev = s
        if prev is not None:
            if prev <= L - 1:
                c += hop_cost(L - 1 - prev, w, 0) + 1 + hop_cost(L - 1, w, 0)
            else:
                c += hop_cost(2 * L - 1 - prev, w, 0)
        tot.append(c)
        kap.append(len(occ))
    return float(np.mean(tot)), float(np.mean(kap))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out = []
    for n, N in ((465, 1024), (512, 1024)):
        for w in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64):
            for m in range(0, 9):
                if m > w:
                    continue
                u, kap = expected_units(n, N, w, m)
                keys = 3 * (n * (1 + m) + w + 2)
                out.append(dict(n=n, N=N, w=w, m=m, units=u, kappa=kap,
                                key_elements=keys, ntt=u * 4))
    with open(os.path.join(RESULTS, "e2_tradeoff.json"), "w") as f:
        json.dump(out, f, indent=1)
    for r in out:
        if r["n"] == 465 and r["m"] in (0, 1, 2, 4, 8) and r["w"] in (1, 4, 16, 64):
            print(f"n={r['n']} w={r['w']:3d} m={r['m']} units={r['units']:8.1f} "
                  f"keys={r['key_elements']:6d} kappa={r['kappa']:6.1f}")


if __name__ == "__main__":
    main()
