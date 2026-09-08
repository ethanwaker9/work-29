import json
import os
import subprocess
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.failure import NoiseModel, noise_weight, ntt_count, EPS_VAR, NTRU_METHODS
from analysis.optimize import min_gadget
from analysis.ntru_fatigue import max_logQ, TERNARY_VAR

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "figures", "out")
PAPER = os.path.abspath(os.path.join(BASE, "..", "final_paper", "figures"))

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.4, "lines.linewidth": 1.3,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

METHODS = ["AP", "GINX", "LMKCDEY", "FINAL", "XZD", "Ours"]
COLORS = {"AP": "#8c8c8c", "GINX": "#4c72b0", "LMKCDEY": "#dd8452",
          "FINAL": "#55a868", "XZD": "#c44e52", "Ours": "#111111"}
HATCH = {"AP": "//", "GINX": "\\\\", "LMKCDEY": "xx", "FINAL": "..",
         "XZD": "++", "Ours": ""}
PARAMNAME = {"PT": "ternary", "PG": "Gaussian", "PTH": "wide"}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PAPER, exist_ok=True)
    eps = os.path.join(OUT, name + ".eps")
    fig.savefig(eps, format="eps")
    plt.close(fig)
    subprocess.run(["epstopdf", eps, "--outfile=" + os.path.join(PAPER, name + ".pdf")],
                   check=True)
    print("wrote", name)


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def fig_time_key(e1):
    idx = {(r["param"], r["method"]): r for r in e1}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.62))
    params = ["PT", "PG", "PTH"]
    x = np.arange(len(params))
    wdt = 0.13
    for k, m in enumerate(METHODS):
        v = [idx[(p, m)]["time_ms"] for p in params]
        axes[0].bar(x + (k - 2.5) * wdt, v, wdt, label=m, color=COLORS[m],
                    hatch=HATCH[m], edgecolor="white", linewidth=0.4)
        v2 = [idx[(p, m)]["brk_MB"] for p in params]
        axes[1].bar(x + (k - 2.5) * wdt, v2, wdt, color=COLORS[m],
                    hatch=HATCH[m], edgecolor="white", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([PARAMNAME[p] for p in params])
    axes[0].set_ylabel("time per gate (ms)")
    axes[0].set_xlabel("LWE secret-key distribution")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([PARAMNAME[p] for p in params])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("blind-rotation key (MB)")
    axes[1].set_xlabel("LWE secret-key distribution")
    axes[0].legend(ncol=3, fontsize=6.5, loc="upper center", framealpha=0.92)
    axes[0].set_ylim(0, 2350)
    save(fig, "time_key")


def fig_ops(e1):
    idx = {(r["param"], r["method"]): r for r in e1}
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    params = ["PT", "PG", "PTH"]
    x = np.arange(len(params))
    wdt = 0.13
    for k, m in enumerate(METHODS):
        v = [idx[(p, m)]["ringmul"] for p in params]
        ax.bar(x + (k - 2.5) * wdt, v, wdt, label=m, color=COLORS[m],
               hatch=HATCH[m], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([PARAMNAME[p] for p in params])
    ax.set_ylabel(r"ring multiplications per gate")
    ax.set_xlabel("LWE secret-key distribution")
    ax.set_yscale("log")
    ax.legend(ncol=2, fontsize=7)
    save(fig, "ops")


def fig_ntru():
    ks = np.arange(0, 8.01, 0.25)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.62))
    for sg, style in ((3.2, "-"), (1.6, "--")):
        lq = [max_logQ(1024, 2.0 ** k, sg) for k in ks]
        axes[0].plot(ks, lq, style, label=r"$\sigma_g=%.1f$" % sg)
    axes[0].axhline(27, color="k", ls=":", lw=1.0)
    axes[0].axhline(19.8, color="#c44e52", ls="-.", lw=1.0)
    axes[0].text(0.1, 27.6, r"$\log_2 Q=27$ (RLWE at $N=1024$)", fontsize=6.5)
    axes[0].text(0.1, 20.4, "symmetric ternary (FINAL, XZD)", fontsize=6.5, color="#c44e52")
    axes[0].set_xlabel(r"$\log_2 \sigma_f$")
    axes[0].set_ylabel(r"admissible $\log_2 Q$ at $128$ bits")
    axes[0].legend(fontsize=7)
    ks2 = np.arange(0, 8.01, 0.5)
    dmin, ntts = [], []
    for k in ks2:
        lq = int(math.floor(max_logQ(1024, 2.0 ** k, 3.2)))
        g = min_gadget(1024, lq, 465, 3.2, 2.0 ** k, math.sqrt(4.0 / 3.0),
                       1.0 / 3.0, 703.0, 1 << 18, 5, -128.0)
        if g is None:
            dmin.append(np.nan)
            ntts.append(np.nan)
        else:
            dmin.append(g[0])
            ntts.append(703.0 * (g[0] + 1))
    axes[1].step(ks2, dmin, where="mid", color="#111111", label="minimal $d$")
    axes[1].set_xlabel(r"$\log_2 \sigma_f$")
    axes[1].set_ylabel(r"minimal $d$ at $p_{\mathrm{fail}}=2^{-128}$")
    axt = axes[1].twinx()
    axt.step(ks2, ntts, where="mid", color="#c44e52", ls="--", label="NTT count")
    axt.set_ylabel("NTT count per gate", color="#c44e52")
    axt.grid(False)
    axes[1].legend(fontsize=7, loc="upper right")
    save(fig, "ntru")


def fig_contract(e3):
    runs = [r for r in e3["runs"] if r["width"] == 8]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.05),
                             gridspec_kw=dict(width_ratios=[2.0, 1.0], wspace=0.42))
    names = [r["method"] for r in runs]
    vals = [r["time_s"] for r in runs]
    order = np.argsort(vals)
    axes[0].barh([names[i] for i in order], [vals[i] for i in order],
                 color=[COLORS[names[i]] for i in order],
                 hatch=[HATCH[names[i]] for i in order], edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("latency of one confidential transfer (s)")
    w32 = [r for r in e3["runs"] if r["width"] == 32]
    if w32:
        axes[1].bar([r["method"] for r in w32], [r["time_s"] for r in w32],
                    color=[COLORS[r["method"]] for r in w32],
                    hatch=[HATCH[r["method"]] for r in w32], edgecolor="white",
                    width=0.55, linewidth=0.4)
        axes[1].set_ylabel("latency, 32-bit (s)")
        axes[1].set_xlabel("method")
    save(fig, "contract")


def fig_scaling(e1):
    idx = {(r["param"], r["method"]): r for r in e1}
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    ns = np.arange(128, 1025, 32)
    N = 1024
    for m, lab in (("GINX", r"GINX, $|U|=6$"), ("FINAL", r"FINAL, $|U|=6$"),
                   ("LMKCDEY", "LMKCDEY"), ("XZD", "XZD"), ("Ours", "Ours")):
        ys = []
        for n in ns:
            kappa = 2 * N * (1.0 - math.exp(-n / (2.0 * N)))
            if m == "GINX":
                y = 6 * n * (2 * 3 + 2)
            elif m == "FINAL":
                y = 6 * n * 4
            elif m == "LMKCDEY":
                y = n * 8 + (kappa + (2 * N - kappa) / 16.0) * 5
            elif m == "XZD":
                y = (2 * n + 1) * 4
            else:
                y = (n + 1 + 0.64 * kappa) * 4
            ys.append(y)
        ax.plot(ns, ys, color=COLORS[m], label=lab)
    ax.set_xlabel("LWE dimension $n$")
    ax.set_ylabel("NTT count per gate")
    ax.legend(fontsize=6.5)
    save(fig, "scaling")


def fig_combined(e1, e2):
    idx = {(r["param"], r["method"]): r for r in e1}
    ter = math.sqrt(TERNARY_VAR)
    fig = plt.figure(figsize=(6.6, 1.68))
    ax0 = fig.add_subplot(1, 3, 1)
    for m in METHODS:
        r = idx[("PG", m)]
        W = noise_weight(m, r["extprod"], r["keyswitch"])
        sring = 32.0 if m in NTRU_METHODS else ter
        xs, ys = [], []
        for d in range(2, 8):
            best = None
            for bb in range(1, 26):
                if d * bb > 27 or 27 - d * bb > 3:
                    continue
                nm = NoiseModel(N=1024, logQ=27, d=d, base_bits=bb, n=r["n"],
                                q_ks=1 << 18, d_ks=5, sigma_g=3.2, sigma_ring=sring,
                                sigma_lwe_key=math.sqrt(4.0 / 3.0), eps_var=EPS_VAR[m],
                                log2_delta=-1200.0)
                lp = nm.log2_pfail(W)
                if best is None or lp < best:
                    best = lp
            if best is None:
                continue
            xs.append(ntt_count(m, r["extprod"], r["keyswitch"], d) / 1000.0)
            ys.append(-best)
        ax0.plot(xs, ys, "o-", ms=2.6, color=COLORS[m], label=m)
    ax0.axhline(128, color="k", ls=":", lw=1.0)
    ax0.text(0.30, 0.05, r"IND-CPA$^{\mathrm{D}}$", fontsize=6.5, transform=ax0.transAxes)
    ax0.set_xlabel(r"NTT count per gate ($10^3$)")
    ax0.set_ylabel(r"$-\log_2$ validated $p_{\mathrm{fail}}$")
    ax0.set_ylim(0, 420)
    ax0.set_xlim(0, 26)
    ax0.legend(ncol=2, fontsize=5.6, loc="upper right", handlelength=1.2)
    rows = [r for r in e2 if r["n"] == 465]
    ws = [8, 12, 16, 24, 32, 48, 64]
    ms = list(range(0, 9))
    Z = np.zeros((len(ms), len(ws)))
    for r in rows:
        if r["w"] in ws and r["m"] in ms:
            Z[ms.index(r["m"]), ws.index(r["w"])] = r["units"]
    W2, M2 = np.meshgrid(np.arange(len(ws)), np.arange(len(ms)))
    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax1.plot_surface(W2, M2, Z, cmap="viridis", edgecolor="k", linewidth=0.2,
                     rstride=1, cstride=1)
    ax1.set_xticks(range(len(ws)))
    ax1.set_xticklabels([str(v) for v in ws], fontsize=5)
    ax1.set_yticks(range(len(ms)))
    ax1.set_yticklabels([str(v) for v in ms], fontsize=5)
    ax1.set_xlabel("window $w$", labelpad=-6, fontsize=7)
    ax1.set_ylabel("depth $m$", labelpad=-6, fontsize=7)
    ax1.set_zlabel("ext. products", labelpad=-6, fontsize=7, rotation=90)
    ax1.tick_params(axis="z", labelsize=5, pad=-1)
    ax1.view_init(elev=24, azim=-128)
    ax1.set_box_aspect((1, 1, 0.85))
    ax2 = fig.add_subplot(1, 3, 3)
    for mm in ms:
        pts = sorted((r["key_elements"] / 1000.0, r["units"]) for r in rows if r["m"] == mm)
        ax2.plot([q[0] for q in pts], [q[1] for q in pts], "o-", ms=2.0,
                 label="$m=%d$" % mm)
    rx = idx[("PG", "XZD")]
    ax2.plot([4.470], [rx["extprod"]], "s", color="#c44e52", ms=5)
    ax2.annotate("XZD", (4.470, rx["extprod"]), textcoords="offset points",
                 xytext=(5, 3), fontsize=6.5, color="#c44e52")
    ax2.set_xlabel(r"key size ($10^3$ elements)")
    ax2.set_ylabel("external products")
    ax2.set_ylim(420, 1550)
    ax2.legend(ncol=3, fontsize=5.2, loc="upper right", handlelength=1.0,
               columnspacing=0.7)
    fig.subplots_adjust(wspace=0.5)
    save(fig, "combined")


def main():
    e1 = load("e1_blindrot.json")
    e2 = load("e2_tradeoff.json")
    fig_time_key(e1)
    fig_ops(e1)
    fig_combined(e1, e2)
    fig_ntru()
    fig_scaling(e1)
    try:
        e3 = load("e3_contract.json")
        fig_contract(e3)
    except FileNotFoundError:
        print("e3 results not available yet")


if __name__ == "__main__":
    main()
