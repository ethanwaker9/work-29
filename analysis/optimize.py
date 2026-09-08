import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.failure import (NoiseModel, noise_weight, ntt_count, ring_mults,
                              key_elements, EPS_VAR, NTRU_METHODS)
from analysis.ntru_fatigue import max_logQ, TERNARY_VAR

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TER = math.sqrt(TERNARY_VAR)
RLWE_LOGQ = 27


def min_gadget(N, logQ, n, sigma_g, sigma_ring, sigma_lwe_key, eps_var, W,
               q_ks, d_ks, target=-128.0, max_shift=3):
    best = None
    for d in range(1, 41):
        for bb in range(1, 26):
            if d * bb > logQ or logQ - d * bb > max_shift:
                continue
            m = NoiseModel(N=N, logQ=logQ, d=d, base_bits=bb, n=n, q_ks=q_ks,
                           d_ks=d_ks, sigma_g=sigma_g, sigma_ring=sigma_ring,
                           sigma_lwe_key=sigma_lwe_key, eps_var=eps_var)
            lp = m.log2_pfail(W)
            if lp <= target and (best is None or lp < best[2]):
                best = (d, bb, lp)
        if best is not None:
            return best
    return None


def load_counts():
    with open(os.path.join(RESULTS, "e1_blindrot.json")) as f:
        data = json.load(f)
    out = {}
    for r in data:
        out[(r["param"], r["method"])] = r
    return out


def main():
    counts = load_counts()
    N = 1024
    q_ks = 1 << 18
    d_ks = 5
    keyvar = {"PT": 2.0 / 3.0, "PG": 4.0 / 3.0, "PTH": 16.0 / 3.0}
    rows = []
    for pname in ("PT", "PG"):
        for meth in ("AP", "GINX", "LMKCDEY", "FINAL", "XZD", "Ours"):
            r = counts[(pname, meth)]
            ext, autos = r["extprod"], r["keyswitch"]
            W = noise_weight(meth, ext, autos)
            sk = math.sqrt(keyvar[pname])
            eps = EPS_VAR[meth]
            fams = [("asym", 32.0, 3.2)] if meth in NTRU_METHODS else [("rlwe", TER, 3.2)]
            if meth in NTRU_METHODS:
                fams.append(("sym", TER, TER))
            for fam, sf, sg in fams:
                if meth in NTRU_METHODS:
                    logQ = int(math.floor(max_logQ(N, sf, sg)))
                    sring, sgn = sf, sg
                else:
                    logQ = RLWE_LOGQ
                    sring, sgn = TER, 3.2
                for target in (-64.0, -128.0):
                    g = min_gadget(N, logQ, r["n"], sgn, sring, sk, eps, W, q_ks, d_ks, target)
                    if g is None:
                        rows.append(dict(param=pname, method=meth, family=fam, target=target,
                                         logQ=logQ, feasible=False))
                        continue
                    d, bb, lp = g
                    rows.append(dict(param=pname, method=meth, family=fam, target=target,
                                     logQ=logQ, d=d, base_bits=bb, log2_pfail=lp,
                                     feasible=True, ext=ext, autos=autos, weight=W,
                                     ntt=ntt_count(meth, ext, autos, d),
                                     ringmul=ring_mults(meth, ext, autos, d),
                                     key_elements=key_elements(meth, r["n"], r["uset"], d, 16)))
    with open(os.path.join(RESULTS, "optimize.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        if not r.get("feasible"):
            print(f"{r['param']:3s} {r['method']:8s} {r['family']:5s} tgt={r['target']:6.0f}  INFEASIBLE")
            continue
        print(f"{r['param']:3s} {r['method']:8s} {r['family']:5s} tgt={r['target']:6.0f} "
              f"logQ={r['logQ']:2d} d={r['d']:2d} bb={r['base_bits']:2d} "
              f"log2p={r['log2_pfail']:8.1f} ntt={r['ntt']:9.0f} mult={r['ringmul']:9.0f} "
              f"keys={r['key_elements']:9d}")


if __name__ == "__main__":
    main()
