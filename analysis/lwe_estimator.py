import math


def delta_bkz(beta):
    if beta <= 2:
        return 1.0219
    return (((math.pi * beta) ** (1.0 / beta)) * beta / (2 * math.pi * math.e)) ** (1.0 / (2.0 * (beta - 1)))


def primal_usvp_beta(n, q, sigma_e, sigma_s=None, m_max=None):
    if sigma_s is None:
        sigma_s = sigma_e
    nu = sigma_e / sigma_s
    best = None
    if m_max is None:
        m_max = 4 * n
    for beta in range(50, 1400):
        dl = delta_bkz(beta)
        ok = False
        for m in range(max(1, n // 2), m_max, max(1, n // 32)):
            d = n + m + 1
            if beta > d:
                continue
            logvol = m * math.log(q) + n * math.log(nu)
            lhs = math.log(math.sqrt(beta / d) * sigma_e * math.sqrt(d))
            rhs = (2 * beta - d - 1) * math.log(dl) + logvol / d
            if lhs <= rhs:
                ok = True
                break
        if ok:
            best = beta
            break
    return best


def core_svp_bits(beta, model="classical"):
    if beta is None:
        return None
    c = 0.292 if model == "classical" else 0.265
    return c * beta


def dual_cost(n, q, sigma_e, sigma_s=None):
    if sigma_s is None:
        sigma_s = sigma_e
    nu = sigma_e / sigma_s
    best = (None, float("inf"))
    lq = math.log(q)
    for beta in range(50, 1400):
        ld = math.log(delta_bkz(beta))
        cbkz = 0.292 * beta
        loc = float("inf")
        for m in range(n + 1, 6 * n, max(1, n // 40)):
            loglen = (m - 1) * ld + n * lq / m
            tau = math.exp(loglen - lq) * sigma_e
            reps = 4 * math.pi ** 2 * tau ** 2 / math.log(2)
            loc = min(loc, cbkz + max(0.0, reps))
        if loc < best[1]:
            best = (beta, loc)
        if cbkz > best[1]:
            break
    return best


def lwe_security(n, q, sigma_e, sigma_s=None):
    b = primal_usvp_beta(n, q, sigma_e, sigma_s)
    bp = core_svp_bits(b)
    bd, cd = dual_cost(n, q, sigma_e, sigma_s)
    c = min(bp, cd)
    return b, c, c * 0.265 / 0.292


def ntru_lambda_from_beta(beta, N):
    if beta is None:
        return None
    return 0.292 * beta + 16.4 + math.log2(8 * N)


if __name__ == "__main__":
    cases = [
        ("FHEW STD128 (n=512,q=1024,sig=3.2,ternary)", 512, 1024, 3.2, math.sqrt(2.0 / 3.0)),
        ("LMKCDEY 128_bGINX (n=571,q=2048,sig=3.2,binary)", 571, 2048, 3.2, 0.5),
        ("Kyber-512 core (n=512,q=3329,sig=1.22)", 512, 3329, 1.22, 1.22),
        ("HES n=1024 logq=27 ternary", 1024, 1 << 27, 3.2, math.sqrt(2.0 / 3.0)),
        ("HES n=2048 logq=54 ternary", 2048, 1 << 54, 3.2, math.sqrt(2.0 / 3.0)),
        ("ours ternary (n=512,q=2^16,sig=205)", 512, 1 << 16, 204.8, math.sqrt(2.0 / 3.0)),
        ("ours gaussian (n=465,q=2^16,sig=205)", 465, 1 << 16, 204.8, math.sqrt(4.0 / 3.0)),
        ("threshold K=8 (n=465,q=2^16,sig=205)", 465, 1 << 16, 204.8, math.sqrt(16.0 / 3.0)),
    ]
    for name, n, q, se, ss in cases:
        b, c, qm = lwe_security(n, q, se, ss)
        print(f"{name:52s} beta={b}  classical={c:.1f}  quantum={qm:.1f}")
