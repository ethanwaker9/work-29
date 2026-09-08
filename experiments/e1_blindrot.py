import sys
import os
import gc
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fhe.params import SETS, METHODS
from fhe.bootstrap import FHEWScheme
from fhe.counters import COUNTER

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def bench(pname, method, trials=30, seed=1234):
    p = SETS[pname]
    rng = np.random.default_rng(seed)
    sch = FHEWScheme(p, method, rng=rng).keygen()
    pairs = [(int(rng.integers(0, 2)), int(rng.integers(0, 2))) for _ in range(trials)]
    cts = [(sch.encrypt(m1), sch.encrypt(m2)) for m1, m2 in pairs]
    ok = 0
    noises = []
    times = []
    COUNTER.reset()
    for (m1, m2), (c1, c2) in zip(pairs, cts):
        if method == "AP":
            a = (-(np.asarray(c1[0]) + np.asarray(c2[0]))) % p.q_ks
            b = (5 * p.q_ks // 8 - c1[1] - c2[1]) % p.q_ks
            sch.rot.preload(a, b, p.q_ks)
        t0 = time.perf_counter()
        out = sch.nand(c1, c2)
        times.append(time.perf_counter() - t0)
        exp = 1 - (m1 & m2)
        ok += int(sch.decrypt(out) == exp)
        noises.append(sch.noise(out, exp))
    c = COUNTER.snapshot()
    brk, ksk = sch.key_bytes()
    res = dict(param=pname, method=method, n=p.n, N=p.N, logQ=p.logQ, d=p.d,
               key_dist=p.key_dist, uset=len(sch.uset), trials=trials, correct=ok,
               time_ms=float(np.median(times)) * 1000.0,
               time_ms_mean=float(np.mean(times)) * 1000.0,
               ntt=c["ntt"] / trials, intt=c["intt"] / trials,
               ringmul=c["ringmul"] / trials,
               extprod=c["extprod"] / trials, keyswitch=c["keyswitch"] / trials,
               units=(c["extprod"] + c["keyswitch"]) / trials,
               brk_elements=sch.rot.key_elements(),
               brk_MB=brk / 1e6, ksk_MB=ksk / 1e6,
               noise_std=float(np.std(noises)), noise_max=float(np.max(np.abs(noises))))
    del sch
    gc.collect()
    return res


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out = []
    for pname in ("PT", "PG", "PTH"):
        for m in METHODS:
            t0 = time.time()
            r = bench(pname, m)
            r["wall_s"] = time.time() - t0
            out.append(r)
            print(f"{pname:4s} {m:8s} t={r['time_ms']:8.1f}ms  mult={r['ringmul']:8.0f} "
                  f"units={r['units']:7.1f} brk={r['brk_MB']:8.2f}MB ok={r['correct']}/{r['trials']} "
                  f"noise={r['noise_std']:8.1f}", flush=True)
    with open(os.path.join(RESULTS, "e1_blindrot.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
