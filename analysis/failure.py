import math

LOG2E = 1.0 / math.log(2.0)


def log2_gauss_tail(x):
    if x <= 0:
        return 0.0
    return LOG2E * (-0.5 * x * x - math.log(x) - 0.5 * math.log(2 * math.pi)) + 1.0


def quantile_for_log2p(target_log2p, lo=1.0, hi=60.0):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if log2_gauss_tail(mid) > target_log2p:
            lo = mid
        else:
            hi = mid
    return hi


def bernstein_slack(M, B, log2_delta):
    lnd = -log2_delta * math.log(2.0)
    var = M * (B ** 4) / 180.0
    cap = (B * B / 4.0) / 3.0
    u = cap * lnd + math.sqrt((cap * lnd) ** 2 + 2.0 * var * lnd)
    mean = M * B * B / 12.0
    return 1.0 + u / mean


class NoiseModel:
    def __init__(self, N, logQ, d, base_bits, n, q_ks, d_ks, sigma_g,
                 sigma_ring, sigma_lwe_key, eps_var, sigma_ks=3.2, t=4,
                 log2_delta=-400.0):
        self.N = N
        self.logQ = logQ
        self.Q = float(2 ** logQ)
        self.d = d
        self.base_bits = base_bits
        self.B = float(2 ** base_bits)
        self.n = n
        self.q_ks = float(q_ks)
        self.d_ks = d_ks
        self.sigma_g = sigma_g
        self.sigma_ks = sigma_ks
        self.sigma_ring = sigma_ring
        self.sigma_lwe_key = sigma_lwe_key
        self.eps_var = eps_var
        self.t = t
        self.log2_delta = log2_delta

    def sigma_unit(self):
        v = self.d * self.N * self.B ** 2 / 12.0 * self.sigma_g ** 2
        shift = max(0.0, self.logQ - self.d * self.base_bits)
        v += self.N * (2.0 ** shift) ** 2 / 12.0 * self.sigma_ring ** 2
        return math.sqrt(v)

    def sigma_acc(self, W, validated=True):
        s = self.sigma_unit() * math.sqrt(W)
        if validated:
            M = W * self.d * self.N
            s *= math.sqrt(bernstein_slack(M, self.B, self.log2_delta))
        return s

    def sigma_out(self, W, validated=True):
        sa = self.sigma_acc(W, validated)
        v1 = (self.q_ks / self.Q) ** 2 * sa ** 2
        v2 = (self.N * self.sigma_ring ** 2 + 1.0) / 12.0
        v3 = self.sigma_ks ** 2 * self.N * self.d_ks
        return math.sqrt(v1 + v2 + v3)

    def sigma_exponent(self, W, validated=True, gates=2):
        so = self.sigma_out(W, validated)
        sc = 2.0 * self.N / self.q_ks
        v = sc ** 2 * gates * so ** 2
        v += self.eps_var * (1.0 + self.n * self.sigma_lwe_key ** 2)
        return math.sqrt(v)

    def margin(self):
        return self.N / float(self.t)

    def log2_pfail(self, W, validated=True, gates=2):
        s = self.sigma_exponent(W, validated, gates)
        p = log2_gauss_tail(self.margin() / s)
        if validated:
            hi = max(p, self.log2_delta)
            lo = min(p, self.log2_delta)
            p = hi + math.log2(1.0 + 2.0 ** (lo - hi))
        return p

    def log2_pfail_worst_mask(self, W, gates=2):
        so = self.sigma_out(W, True)
        sc = 2.0 * self.N / self.q_ks
        s = math.sqrt(sc ** 2 * gates * so ** 2)
        c = math.sqrt(3.0 * self.eps_var)
        offset = c * self.n * self.sigma_lwe_key * math.sqrt(2.0 / math.pi)
        m = self.margin() - offset
        if m <= 0:
            return 0.0
        return log2_gauss_tail(m / s)


def noise_weight(method, ext, autos):
    if method == "AP":
        return 2.0 * ext
    if method == "GINX":
        return 4.0 * ext
    if method == "LMKCDEY":
        return 2.0 * ext + autos
    if method == "FINAL":
        return 2.0 * ext
    return float(ext)


def ntt_count(method, ext, autos, d):
    if method in ("AP", "GINX"):
        return ext * (2 * d + 2)
    if method == "LMKCDEY":
        return ext * (2 * d + 2) + autos * (d + 2)
    return ext * (d + 1)


def ring_mults(method, ext, autos, d):
    if method in ("AP", "GINX"):
        return ext * 4 * d
    if method == "LMKCDEY":
        return ext * 4 * d + autos * 2 * d
    return ext * d


def key_elements(method, n, uset, d, w, dr=2, Br=32):
    if method == "AP":
        return 4 * d * n * dr * (Br - 1)
    if method == "GINX":
        return 4 * d * n * uset
    if method == "LMKCDEY":
        return 4 * d * n + 2 * d * (w + 1)
    if method == "FINAL":
        return d * (n * uset + 1)
    if method == "XZD":
        return d * (n + 2 + 1023)
    return d * (2 * n + w + 2)


EPS_VAR = {"AP": 1.0 / 3.0, "GINX": 1.0 / 12.0, "LMKCDEY": 1.0 / 3.0,
           "FINAL": 1.0 / 12.0, "XZD": 1.0 / 3.0, "Ours": 1.0 / 3.0}
NTRU_METHODS = ("FINAL", "XZD", "Ours")


if __name__ == "__main__":
    import math as _m
    cases = [("Ours", 704), ("XZD", 930), ("LMKCDEY", 1305), ("AP", 1800),
             ("FINAL", 5574), ("GINX", 11143)]
    for name, W in cases:
        ntru = name in NTRU_METHODS
        m = NoiseModel(N=1024, logQ=27, d=3, base_bits=8, n=465, q_ks=1 << 18,
                       d_ks=5, sigma_g=3.2,
                       sigma_ring=(32.0 if ntru else _m.sqrt(2.0 / 3.0)),
                       sigma_lwe_key=_m.sqrt(4.0 / 3.0),
                       eps_var=EPS_VAR[name])
        h = m.log2_pfail(W, validated=False)
        c = m.log2_pfail(W)
        print(f"{name:8s} W={W:6d} sigma_E={m.sigma_exponent(W):7.2f} "
              f"heuristic={h:9.2f} validated={c:9.2f} loss={c - h:5.2f}")
    print("Bernstein slack gamma =", bernstein_slack(704 * 3 * 1024, 256.0, -400.0) - 1.0)
