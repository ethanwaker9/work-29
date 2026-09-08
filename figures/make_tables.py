import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.failure import NoiseModel, noise_weight, ntt_count, EPS_VAR, NTRU_METHODS
from analysis.ntru_fatigue import TERNARY_VAR

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "results", "tables.txt")
METHODS = ["AP", "GINX", "LMKCDEY", "FINAL", "XZD", "Ours"]
KEYVAR = {"PT": 2.0 / 3.0, "PG": 4.0 / 3.0, "PTH": 16.0 / 3.0}
TER = math.sqrt(TERNARY_VAR)


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def validated(r):
    m = r["method"]
    W = noise_weight(m, r["extprod"], r["keyswitch"])
    ntru = m in NTRU_METHODS
    nm = NoiseModel(N=1024, logQ=27, d=3, base_bits=8, n=r["n"], q_ks=1 << 18, d_ks=5,
                    sigma_g=3.2, sigma_ring=(32.0 if ntru else TER),
                    sigma_lwe_key=math.sqrt(KEYVAR[r["param"]]), eps_var=EPS_VAR[m])
    return W, nm


def main():
    e1 = load("e1_blindrot.json")
    idx = {(r["param"], r["method"]): r for r in e1}
    lines = []
    lines.append("%% ===== tab:overview =====")
    for m in METHODS:
        r = idx[("PG", m)]
        W, nm = validated(r)
        lp = nm.log2_pfail(W)
        lines.append("%s & %d & %.0f & %.2f & %.1f & %s \\\\" %
                     (m, r["n"], r["ringmul"], r["brk_MB"], r["time_ms"],
                      "yes" if lp <= -128 else "no"))
    lines.append("%% ===== tab:micro =====")
    for p in ("PT", "PG"):
        for m in METHODS:
            r = idx[(p, m)]
            W, nm = validated(r)
            lp = nm.log2_pfail(W)
            lines.append(" & %s & %.0f & %.0f & %.2f & %.2f & %.1f & $%.1f$ \\\\" %
                         (m, r["ringmul"], r["ntt"] + r["intt"], r["brk_MB"],
                          r["ksk_MB"], r["time_ms"], lp))
    lines.append("%% ===== tab:wide (PTH) =====")
    for m in METHODS:
        r = idx[("PTH", m)]
        lines.append("%s & %.0f & %.2f & %.1f & %.1f \\\\" %
                     (m, r["ringmul"], r["brk_MB"], r["time_ms"], r["noise_std"]))
    lines.append("%% ===== tab:validated (PG) =====")
    for m in ["Ours", "XZD", "LMKCDEY", "AP", "FINAL", "GINX"]:
        r = idx[("PG", m)]
        W, nm = validated(r)
        h = nm.log2_pfail(W, validated=False)
        c = nm.log2_pfail(W)
        lines.append("%s & %.0f & %.2f & $%.2f$ & $%.2f$ & %.2f \\\\" %
                     (m, W, nm.sigma_exponent(W), h, c, c - h))
    lines.append("%% slack gamma = %.6f" % (
        __import__("analysis.failure", fromlist=["bernstein_slack"]).bernstein_slack(
            703 * 3 * 1024, 256.0, -400.0) - 1.0))
    try:
        e3 = load("e3_contract.json")
        lines.append("%% ===== tab:contract =====")
        for r in e3["runs"]:
            if r["width"] != 8:
                continue
            lines.append("%s & %d & %d & %.1f & %.1f & %.2f \\\\" %
                         (r["method"], r["gates"], r["ntt"], r["time_s"],
                          r["time_per_gate_ms"], r["brk_MB"]))
        lines.append("%% 32-bit runs")
        for r in e3["runs"]:
            if r["width"] == 32:
                lines.append("%% %s gates=%d time=%.1f s" % (r["method"], r["gates"], r["time_s"]))
        lines.append("%% gas: " + json.dumps(e3["gas"]))
    except FileNotFoundError:
        lines.append("%% e3 not available")
    txt = "\n".join(lines)
    with open(OUT, "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
