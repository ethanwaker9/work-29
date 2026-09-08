import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fhe.ring import Ring, Gadget, find_ntt_prime, sample_secret
from fhe.walk import ms_alpha
from fhe.blindrot import (APRotator, GINXRotator, LMKRotator, NgsGinxRotator,
                          XZDRotator, FusedRotator, gadget_u_set, monomial)
from fhe.counters import COUNTER


def run(N=128, logQ=27, d=3, bb=9, n=12, q_ks=1 << 14, dist="ternary", seed=7):
    Q = find_ntt_prime(N, logQ)
    R = Ring(N, Q)
    G = Gadget(R, bb, d)
    rng = np.random.default_rng(seed)
    s = sample_secret(rng, dist, n)
    a = rng.integers(0, q_ks, n, dtype=np.int64)
    b = int(rng.integers(0, q_ks))
    tv = np.zeros(N, dtype=np.int64)
    tv[0] = Q // 8
    ub = int(max(1, np.max(np.abs(s))))
    uset = gadget_u_set("ternary" if dist == "ternary" else ("binary" if dist == "binary" else "g"), ub)
    rots = [APRotator(R, G, 3.2, rng, n, q_ks, 32),
            GINXRotator(R, G, 3.2, rng, n, q_ks, uset),
            LMKRotator(R, G, 3.2, rng, n, q_ks, window=4),
            NgsGinxRotator(R, G, 3.2, rng, n, q_ks, uset),
            XZDRotator(R, G, 3.2, rng, n, q_ks),
            FusedRotator(R, G, 3.2, rng, n, q_ks, window=4, fuse=2)]
    ok = True
    for rot in rots:
        COUNTER.reset()
        rot.keygen(s)
        acc = rot.rotate(a, b, q_ks, tv)
        al, be = ms_alpha(a, b, q_ks, 2 * N, rot.ms_mode)
        mu = (be + int(np.dot(al, s))) % (2 * N)
        target = R.center(monomial(R, mu) * (Q // 8) % Q)
        ph = rot.ctx.phase(acc)
        err = int(np.max(np.abs(ph - target)))
        good = err < Q // 64
        ok &= good
        print(f"{rot.name:9s} {'OK ' if good else 'BAD'} maxerr={err:9d} "
              f"mults={COUNTER.ringmul:6d} ntt={COUNTER.ntt:6d} keys={rot.key_elements()}")
    return ok


if __name__ == "__main__":
    res = []
    for kw in (dict(dist="ternary"), dict(dist="gaussian"), dict(N=256, n=40, dist="ternary"),
               dict(N=256, n=40, dist="gaussian", seed=11)):
        print("===", kw, "===")
        res.append(run(**kw))
    print("ALL OK" if all(res) else "FAILURES")
