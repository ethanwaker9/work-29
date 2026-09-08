import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.lwe_estimator import delta_bkz

TERNARY_VAR = 2.0 / 3.0
DVW_TERNARY_CONST = 0.004
DVW_EXPONENT = 2.484
UNIT_CONST = DVW_TERNARY_CONST / TERNARY_VAR
ANCHOR_LOGQ_128 = {1024: 19.8}


def effective_modulus(Q, sigma_f, sigma_g):
    return Q / (sigma_f * sigma_g)


def fatigue_point(N, sigma_f=None, sigma_g=None):
    if sigma_f is None:
        sigma_f = math.sqrt(TERNARY_VAR)
    if sigma_g is None:
        sigma_g = sigma_f
    return UNIT_CONST * sigma_f * sigma_g * (N ** DVW_EXPONENT)


def balance(N, Q, sigma_f, sigma_g):
    c = sigma_g / sigma_f
    return (N, Q * c, sigma_f * c, sigma_g)


def invariants(N, Q, sigma_f, sigma_g):
    dim = 2 * N
    log_vol = N * math.log(Q)
    target = math.sqrt(N * (sigma_f ** 2 + sigma_g ** 2))
    dsl_log_vol = N * 0.5 * math.log(N * (sigma_f ** 2 + sigma_g ** 2))
    gh = math.sqrt(dim / (2 * math.pi * math.e)) * math.exp(log_vol / dim)
    return dict(dim=dim,
                unit_log_vol=log_vol / dim - math.log(sigma_g),
                unit_target=target / sigma_g,
                unit_dsl=dsl_log_vol / N - math.log(sigma_g),
                ratio=target / gh)


def skr_beta(N, Q, sigma_f, sigma_g, balanced=True):
    if balanced:
        N, Q, sigma_f, sigma_g = balance(N, Q, sigma_f, sigma_g)
    dim = 2 * N
    target = math.sqrt(N * (sigma_f ** 2 + sigma_g ** 2))
    lv = N * math.log(Q) / dim
    for beta in range(40, 4 * N):
        dl = math.log(delta_bkz(beta))
        lhs = math.log(math.sqrt(beta / dim) * target)
        rhs = (2 * beta - dim - 1) * dl + lv
        if lhs <= rhs:
            return beta
    return None


def bkz_bits(beta, N, model="practical"):
    if beta is None:
        return None
    if model == "core":
        return 0.292 * beta
    return 0.292 * beta + 16.4 + math.log2(16 * N)


def max_logQ(N, sigma_f, sigma_g, anchor=None):
    if anchor is None:
        anchor = ANCHOR_LOGQ_128.get(N)
        if anchor is None:
            raise ValueError("no anchor for N=%d" % N)
    return anchor + math.log2(sigma_f * sigma_g / TERNARY_VAR)


def gadget_dimension(logQ, base_bits):
    return int(math.ceil(logQ / float(base_bits)))


def sweep(N=1024, sigma_g=3.2, base_bits=8):
    out = []
    for k in range(0, 8):
        sf = 2.0 ** k
        lq = max_logQ(N, sf, sigma_g)
        d = gadget_dimension(lq, base_bits)
        out.append(dict(log2_sigma_f=k, sigma_f=sf, logQmax=lq, d=d,
                        ntt_per_unit=d + 1,
                        key_elements_per_unit=d))
    return out


if __name__ == "__main__":
    ter = math.sqrt(TERNARY_VAR)
    print("Invariance of the balanced image (N=1024, Q=2^27)")
    for (sf, sg) in ((32.0, 3.2), (8.0, 12.8), (64.0, 1.6), (102.4, 1.0)):
        Q = 2.0 ** 27
        b = invariants(*balance(1024, Q, sf, sg))
        print(f"  sf={sf:7.3f} sg={sg:6.3f} Qeff=2^{math.log2(effective_modulus(Q, sf, sg)):.4f} "
              f"unit_vol={b['unit_log_vol']:.6f} unit_target={b['unit_target']:.4f} "
              f"unit_dsl={b['unit_dsl']:.6f} beta_skr={skr_beta(1024, Q, sf, sg)}")
    print()
    print("Naive (unbalanced) versus balanced SKR blocksize")
    for (sf, sg) in ((32.0, 3.2), (8.0, 12.8), (64.0, 1.6)):
        Q = 2.0 ** 27
        print(f"  sf={sf:6.2f} sg={sg:6.2f} naive={skr_beta(1024, Q, sf, sg, False)} "
              f"balanced={skr_beta(1024, Q, sf, sg)}")
    print()
    print("Fatigue point by transport")
    for N in (512, 1024, 2048):
        print(f"  N={N:5d}: ternary 2^{math.log2(fatigue_point(N)):.2f}, "
              f"(32, 3.2) 2^{math.log2(fatigue_point(N, 32.0, 3.2)):.2f}")
    print()
    print("Admissible log2 Q at 128 bits, N=1024 (anchored at ternary 19.8)")
    for row in sweep():
        print(f"  sigma_f=2^{row['log2_sigma_f']:<2d} logQmax={row['logQmax']:6.2f} "
              f"d={row['d']} ntt/unit={row['ntt_per_unit']}")
