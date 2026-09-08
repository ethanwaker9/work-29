import json
import os
import sys
import time
import gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fhe.params import SETS, METHODS
from fhe.bootstrap import FHEWScheme
from fhe.counters import COUNTER
from contracts.circuits import Evaluator, GateCounter, ripple_add, ripple_sub, geq, mux_word
from contracts.gas import transfer_cost, ct_bytes

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def enc_word(sch, value, width):
    return [sch.encrypt((value >> i) & 1) for i in range(width)]


def dec_word(sch, bits):
    v = 0
    for i, b in enumerate(bits):
        v |= (sch.decrypt(b) & 1) << i
    return v


def transfer(ev, sch, bal_from, bal_to, amount, width):
    ok = geq(ev, bal_from, amount)
    zero = enc_word(sch, 0, width)
    amt = mux_word(ev, ok, amount, zero)
    nf, _ = ripple_sub(ev, bal_from, amt)
    nt, _ = ripple_add(ev, bal_to, amt)
    return ok, nf, nt


def run(pname, method, width=8, seed=99, verbose=True):
    p = SETS[pname]
    rng = np.random.default_rng(seed)
    sch = FHEWScheme(p, method, rng=rng).keygen()
    pre = (lambda a, b: sch.rot.preload(a, b, p.q_ks)) if method == "AP" else None
    ev = Evaluator(sch, GateCounter(), pre)
    mask = (1 << width) - 1
    v_from, v_to, v_amt = 200 & mask, 37 & mask, 145 & mask
    bf = enc_word(sch, v_from, width)
    bt = enc_word(sch, v_to, width)
    am = enc_word(sch, v_amt, width)
    COUNTER.reset()
    ok, nf, nt = transfer(ev, sch, bf, bt, am, width)
    dt = ev.c.time
    got_ok = sch.decrypt(ok)
    got_f = dec_word(sch, nf)
    got_t = dec_word(sch, nt)
    exp_ok = 1 if v_from >= v_amt else 0
    exp_f = (v_from - (v_amt if exp_ok else 0)) % (1 << width)
    exp_t = (v_to + (v_amt if exp_ok else 0)) % (1 << width)
    correct = (got_ok == exp_ok) and (got_f == exp_f) and (got_t == exp_t)
    brk, ksk = sch.key_bytes()
    res = dict(param=pname, method=method, width=width, gates=ev.c.count,
               free_gates=ev.c.free, time_s=dt, time_per_gate_ms=dt / ev.c.count * 1000,
               correct=bool(correct), got=(got_ok, got_f, got_t), exp=(exp_ok, exp_f, exp_t),
               ringmul=COUNTER.ringmul, ntt=COUNTER.ntt + COUNTER.intt,
               brk_MB=brk / 1e6, ksk_MB=ksk / 1e6,
               ct_bytes=ct_bytes(p.n, int(np.log2(p.q_ks))))
    if verbose:
        print(f"{pname:3s} {method:8s} w={width} gates={res['gates']:5d} t={dt:8.2f}s "
              f"({res['time_per_gate_ms']:7.1f} ms/gate) correct={correct}", flush=True)
    del sch
    gc.collect()
    return res


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out = []
    for m in METHODS:
        out.append(run("PG", m, width=8))
    out.append(run("PG", "Ours", width=32))
    out.append(run("PG", "XZD", width=32))
    p = SETS["PG"]
    gas = [transfer_cost(p.n, 18, "onchain"), transfer_cost(p.n, 18, "handle")]
    with open(os.path.join(RESULTS, "e3_contract.json"), "w") as f:
        json.dump(dict(runs=out, gas=gas), f, indent=1)
    for g in gas:
        print(g)


if __name__ == "__main__":
    main()
