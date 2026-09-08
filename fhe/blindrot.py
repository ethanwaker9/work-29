import numpy as np
from .ring import Gadget
from .rlwe import RlweCtx
from .ngs import NgsCtx
from .counters import COUNTER
from .walk import build_stops, build_stops_sub, hop_split, ms_alpha


def monomial(ring, e):
    N = ring.n
    e = int(e) % (2 * N)
    p = np.zeros(N, dtype=np.int64)
    if e < N:
        p[e] = 1
    else:
        p[e - N] = ring.q - 1
    return p


def mul_monomial(ring, a, e):
    N = ring.n
    e = int(e) % (2 * N)
    s = 1
    if e >= N:
        e -= N
        s = -1
    out = np.roll(a, e, axis=-1)
    if e:
        out[..., :e] = -out[..., :e]
    if s < 0:
        out = -out
    return out % ring.q


def signed_digits(x, base, d):
    out = []
    for _ in range(d):
        out.append(x % base)
        x //= base
    return out


def gadget_u_set(dist, bound):
    if dist == "binary":
        return [1]
    if dist == "ternary":
        return [1, -1]
    k = int(np.ceil(np.log2(bound + 1)))
    u = []
    for j in range(k):
        u.append(1 << j)
        u.append(-(1 << j))
    return u


def decompose_secret(s, uset):
    n = len(s)
    out = np.zeros((len(uset), n), dtype=np.int64)
    if uset == [1]:
        out[0] = s
        return out
    if uset == [1, -1]:
        out[0] = (s == 1).astype(np.int64)
        out[1] = (s == -1).astype(np.int64)
        return out
    pos = np.array([u for u in uset])
    for i in range(n):
        v = int(s[i])
        sign = 1 if v >= 0 else -1
        v = abs(v)
        for j, u in enumerate(uset):
            if u * sign <= 0:
                continue
            bit = (v >> int(np.log2(abs(u)))) & 1
            out[j, i] = bit
    return out


class Baseline:
    def ring_secret(self):
        raise NotImplementedError


class APRotator(Baseline):
    name = "AP"
    ms_mode = "half"

    def __init__(self, ring, gadget, sigma, rng, n, q, base_r, key_dist="ternary"):
        self.R = ring
        self.ctx = RlweCtx(ring, gadget, sigma, rng)
        self.n = n
        self.q = q
        self.nu = 2 * ring.n // q
        self.Br = base_r
        self.dr = int(np.ceil(np.log(ring.n) / np.log(base_r)))
        self.key_dist = key_dist

    def keygen(self, s, z_dist="ternary"):
        self.ctx.keygen(z_dist)
        self.brk = {}
        self.s = s

    def _key(self, i, j, v):
        k = (i, j, v)
        c = self.brk.get(k)
        if c is None:
            e = 2 * v * (self.Br ** j) * int(self.s[i])
            c = self.ctx.prep_rgsw(self.ctx.rgsw(monomial(self.R, e)))
            self.brk[k] = c
        return c

    def preload(self, a, b, q_ks):
        self.brk = {}
        alpha, _ = ms_alpha(a, b, q_ks, 2 * self.R.n, "half")
        for i in range(self.n):
            for j, v in enumerate(signed_digits(int(alpha[i]) // 2, self.Br, self.dr)):
                if v:
                    self._key(i, j, v)

    def key_elements(self):
        return 4 * self.ctx.G.d * self.n * self.dr * (self.Br - 1)

    def rotate(self, a, b, q_ks, tv):
        R = self.R
        alpha, beta = ms_alpha(a, b, q_ks, 2 * R.n, "half")
        acc = np.stack([np.zeros(R.n, dtype=np.int64), mul_monomial(R, tv, beta)])
        for i in range(self.n):
            dig = signed_digits(int(alpha[i]) // 2, self.Br, self.dr)
            for j, v in enumerate(dig):
                if v == 0:
                    continue
                acc = self.ctx.ext_prod(acc, self._key(i, j, v))
        return acc

    def ring_secret(self):
        return self.ctx.z_c


class GINXRotator(Baseline):
    name = "GINX"
    ms_mode = "plain"

    def __init__(self, ring, gadget, sigma, rng, n, q, uset, key_dist="binary"):
        self.R = ring
        self.ctx = RlweCtx(ring, gadget, sigma, rng)
        self.n = n
        self.q = q
        self.nu = 2 * ring.n // q
        self.uset = uset
        self.key_dist = key_dist

    def keygen(self, s, z_dist="ternary"):
        self.ctx.keygen(z_dist)
        R = self.R
        bits = decompose_secret(s, self.uset)
        self.brk = []
        for i in range(self.n):
            row = []
            for j in range(len(self.uset)):
                m = np.zeros(R.n, dtype=np.int64)
                m[0] = int(bits[j, i])
                row.append(self.ctx.prep_rgsw(self.ctx.rgsw(m)))
            self.brk.append(row)

    def key_elements(self):
        return 4 * self.ctx.G.d * self.n * len(self.uset)

    def rotate(self, a, b, q_ks, tv):
        R = self.R
        Q = R.q
        alpha, beta = ms_alpha(a, b, q_ks, 2 * R.n, "plain")
        acc = np.stack([np.zeros(R.n, dtype=np.int64), mul_monomial(R, tv, beta)])
        for i in range(self.n):
            for j, u in enumerate(self.uset):
                e = (u * int(alpha[i])) % (2 * R.n)
                if e == 0:
                    continue
                t = self.ctx.ext_prod(acc, self.brk[i][j])
                acc = (acc + mul_monomial(R, t, e) - t) % Q
        return acc

    def ring_secret(self):
        return self.ctx.z_c


class LMKRotator(Baseline):
    name = "LMKCDEY"
    ms_mode = "odd"

    def __init__(self, ring, gadget, sigma, rng, n, q, window=10, g=5):
        self.R = ring
        self.ctx = RlweCtx(ring, gadget, sigma, rng)
        self.n = n
        self.q = q
        self.nu = 2 * ring.n // q
        self.w = window
        self.g = g

    def keygen(self, s, z_dist="ternary"):
        self.ctx.keygen(z_dist)
        R = self.R
        two_n = 2 * R.n
        self.brk = [self.ctx.prep_rgsw(self.ctx.rgsw(monomial(R, int(s[i])))) for i in range(self.n)]
        self.ak = {}
        for u in range(1, self.w + 1):
            t = pow(self.g, u, two_n)
            self.ak[t] = self.ctx.prep_rlwep(self.ctx.auto_key(t))
        tneg = (-self.g) % two_n
        self.ak[tneg] = self.ctx.prep_rlwep(self.ctx.auto_key(tneg))

    def key_elements(self):
        d = self.ctx.G.d
        return 4 * d * self.n + 2 * d * (self.w + 1)

    def rotate(self, a, b, q_ks, tv):
        R = self.R
        two_n = 2 * R.n
        alpha, beta = ms_alpha(a, b, q_ks, two_n, "odd")
        act = [i for i in range(self.n) if alpha[i] != 0]
        wodd = alpha
        buckets, L = build_stops_sub(wodd, act, two_n, self.g)
        t0 = (-self.g) % two_n
        p0 = mul_monomial(R, R.automorphism(tv, t0), (t0 * beta) % two_n)
        acc = np.stack([np.zeros(R.n, dtype=np.int64), p0])
        cur = None
        for stop in range(2 * L):
            if not buckets[stop]:
                continue
            if cur is None:
                if stop > 0:
                    t = pow(self.g, stop, two_n)
                    if 0 <= L - 1 < stop:
                        t = (-t) % two_n
                    acc = np.stack([acc[0], R.automorphism(acc[1], t)])
            if cur is not None:
                delta = stop - cur
                cross = (cur <= L - 1 < stop)
                if cross:
                    d1 = (L - 1) - cur
                    if d1 > 0:
                        acc = self._shift(acc, d1)
                    acc = self._auto(acc, (-self.g) % two_n)
                    d2 = stop - L
                    if d2 > 0:
                        acc = self._shift(acc, d2)
                else:
                    acc = self._shift(acc, delta)
            cur = stop
            for i in buckets[stop]:
                acc = self.ctx.ext_prod(acc, self.brk[i])
        if cur is not None and cur <= L - 1:
            d1 = (L - 1) - cur
            if d1 > 0:
                acc = self._shift(acc, d1)
            acc = self._auto(acc, (-self.g) % two_n)
            if L - 1 > 0:
                acc = self._shift(acc, L - 1)
        elif cur is not None:
            d2 = (2 * L - 1) - cur
            if d2 > 0:
                acc = self._shift(acc, d2)
        return acc

    def _auto(self, acc, t):
        return self.ctx.apply_auto(acc, t, self.ak[t])

    def _shift(self, acc, delta):
        two_n = 2 * self.R.n
        while delta > 0:
            h = min(self.w, delta)
            acc = self._auto(acc, pow(self.g, h, two_n))
            delta -= h
        return acc

    def ring_secret(self):
        return self.ctx.z_c


class NgsGinxRotator(Baseline):
    name = "FINAL"
    ms_mode = "plain"

    def __init__(self, ring, gadget, sigma_g, rng, n, q, uset, sigma_f=None):
        self.R = ring
        self.ctx = NgsCtx(ring, gadget, sigma_g, rng, sigma_f)
        self.n = n
        self.q = q
        self.nu = 2 * ring.n // q
        self.uset = uset

    def keygen(self, s):
        self.ctx.keygen()
        R = self.R
        bits = decompose_secret(s, self.uset)
        self.brk = []
        for i in range(self.n):
            row = []
            for j in range(len(self.uset)):
                m = np.zeros(R.n, dtype=np.int64)
                m[0] = int(bits[j, i])
                row.append(self.ctx.prep(self.ctx.vector_ct(m)))
            self.brk.append(row)
        self.ini = self.ctx.prep(self.ctx.vector_ct(self.ctx.finv))

    def key_elements(self):
        d = self.ctx.G.d
        return d * (self.n * len(self.uset) + 1)

    def rotate(self, a, b, q_ks, tv):
        R = self.R
        Q = R.q
        alpha, beta = ms_alpha(a, b, q_ks, 2 * R.n, "plain")
        acc = self.ctx.ext_prod(mul_monomial(R, tv, beta), self.ini)
        for i in range(self.n):
            for j, u in enumerate(self.uset):
                e = (u * int(alpha[i])) % (2 * R.n)
                if e == 0:
                    continue
                t = self.ctx.ext_prod(acc, self.brk[i][j])
                acc = (acc + mul_monomial(R, t, e) - t) % Q
        return acc

    def ring_secret(self):
        return self.ctx.f_c


class XZDRotator(Baseline):
    name = "XZD"
    ms_mode = "half"

    def __init__(self, ring, gadget, sigma_g, rng, n, q, sigma_f=None):
        self.R = ring
        self.ctx = NgsCtx(ring, gadget, sigma_g, rng, sigma_f)
        self.n = n
        self.q = q
        self.nu = 2 * ring.n // q

    def keygen(self, s):
        self.ctx.keygen()
        R = self.R
        two_n = 2 * R.n
        self.evk = [self.ctx.prep(self.ctx.vector_ct(monomial(R, int(s[i])))) for i in range(self.n)]
        self.evk0 = self.ctx.prep(self.ctx.vector_ct(R.mul(monomial(R, int(s[0])), self.ctx.finv)))
        self.nsum = self.ctx.prep(self.ctx.vector_ct(monomial(R, -int(np.sum(s)))))
        self.ksk = {}
        finv_hat = R.ntt(self.ctx.finv)
        for t in range(1, two_n, 2):
            if t == 1:
                continue
            ft = R.automorphism(self.ctx.f, t)
            v = R.intt(R.ntt(ft) * finv_hat % R.q)
            self.ksk[t] = self.ctx.prep(self.ctx.vector_ct(v))
        self.s = s

    def key_elements(self):
        d = self.ctx.G.d
        return d * (self.n + 2 + len(self.ksk))

    def rotate(self, a, b, q_ks, tv):
        R = self.R
        two_n = 2 * R.n
        alpha, beta = ms_alpha(a, b, q_ks, two_n, "half")
        wodd = [(int(alpha[i]) + 1) % two_n for i in range(self.n)]
        winv = [pow(x, -1, two_n) for x in wodd]
        p = mul_monomial(R, R.automorphism(tv, winv[0]), (winv[0] * beta) % two_n)
        acc = self.ctx.ext_prod(p, self.evk0)
        for i in range(self.n):
            if i > 0:
                acc = self.ctx.ext_prod(acc, self.evk[i])
            nxt = winv[i + 1] if i + 1 < self.n else 1
            t = wodd[i] * nxt % two_n
            if t != 1:
                acc = R.automorphism(acc, t)
                COUNTER.automorph += 1
                acc = self.ctx.ext_prod(acc, self.ksk[t])
        acc = self.ctx.ext_prod(acc, self.nsum)
        return acc

    def ring_secret(self):
        return self.ctx.f_c


class FusedRotator(Baseline):
    name = "Ours"
    ms_mode = "odd"

    def __init__(self, ring, gadget, sigma_g, rng, n, q, window=10, fuse=1, g=5, sigma_f=None):
        self.R = ring
        self.ctx = NgsCtx(ring, gadget, sigma_g, rng, sigma_f)
        self.n = n
        self.q = q
        self.nu = 2 * ring.n // q
        self.w = window
        self.m = fuse
        self.g = g

    def keygen(self, s):
        self.ctx.keygen()
        R = self.R
        Q = R.q
        two_n = 2 * R.n
        finv_hat = R.ntt(self.ctx.finv)
        self.evk = [self.ctx.prep(self.ctx.vector_ct(monomial(R, int(s[i])))) for i in range(self.n)]
        self.ratio = {}
        for u in range(1, max(self.w, self.m) + 1):
            t = pow(self.g, u, two_n)
            ft = R.automorphism(self.ctx.f, t)
            self.ratio[u] = R.intt(R.ntt(ft) * finv_hat % Q)
        self.fused = []
        for i in range(self.n):
            row = {}
            mono = monomial(R, int(s[i]))
            for u in range(1, self.m + 1):
                v = R.mul(mono, self.ratio[u])
                row[u] = self.ctx.prep(self.ctx.vector_ct(v))
            self.fused.append(row)
        self.ak = {}
        for u in range(1, self.w + 1):
            self.ak[u] = self.ctx.prep(self.ctx.vector_ct(self.ratio[u]))
        tneg = (-self.g) % two_n
        fneg = R.automorphism(self.ctx.f, tneg)
        self.ak_neg = self.ctx.prep(self.ctx.vector_ct(R.intt(R.ntt(fneg) * finv_hat % Q)))
        self.ini = self.ctx.prep(self.ctx.vector_ct(self.ctx.finv))

    def key_elements(self):
        d = self.ctx.G.d
        return d * (self.n * (1 + self.m) + self.w + 2)

    def rotate(self, a, b, q_ks, tv):
        R = self.R
        two_n = 2 * R.n
        alpha, beta = ms_alpha(a, b, q_ks, two_n, "odd")
        act = [i for i in range(self.n) if alpha[i] != 0]
        buckets, L = build_stops_sub(alpha, act, two_n, self.g)
        t0 = (-self.g) % two_n
        acc = mul_monomial(R, R.automorphism(tv, t0), (t0 * beta) % two_n)
        started = False
        cur = None
        for stop in range(2 * L):
            idx = buckets[stop]
            if not idx:
                continue
            if not started:
                if cur is None and stop > 0:
                    acc = self._perm_plain(acc, stop, L)
                elif cur is not None:
                    acc = self._perm_plain_from(acc, cur, stop, L)
                acc = self.ctx.ext_prod(acc, self.ini)
                for i in idx:
                    acc = self.ctx.ext_prod(acc, self.evk[i])
                started = True
                cur = stop
                continue
            acc, fused_i = self._travel(acc, cur, stop, idx, L)
            for i in idx:
                if i == fused_i:
                    continue
                acc = self.ctx.ext_prod(acc, self.evk[i])
            cur = stop
        if not started:
            acc = self.ctx.ext_prod(acc, self.ini)
            cur = 0
        acc = self._finish(acc, cur, L)
        return acc

    def _perm_plain(self, acc, stop, L):
        return self._perm_plain_from(acc, 0, stop, L, first=True)

    def _perm_plain_from(self, acc, cur, stop, L, first=False):
        R = self.R
        two_n = 2 * R.n
        start = 0 if first else cur
        d = stop - start
        cross = (start <= L - 1 < stop)
        t = pow(self.g, d, two_n)
        if cross:
            t = (-t) % two_n
        if t != 1:
            acc = R.automorphism(acc, t)
        return acc

    def _travel(self, acc, cur, stop, idx, L):
        R = self.R
        two_n = 2 * R.n
        cross = (cur <= L - 1 < stop)
        fused_i = None
        if cross:
            d1 = (L - 1) - cur
            if d1 > 0:
                acc = self._hops(acc, d1, None)[0]
            acc = R.automorphism(acc, (-self.g) % two_n)
            COUNTER.automorph += 1
            acc = self.ctx.ext_prod(acc, self.ak_neg)
            d2 = stop - L
            if d2 > 0:
                acc, fused_i = self._hops(acc, d2, idx)
        else:
            acc, fused_i = self._hops(acc, stop - cur, idx)
        return acc, fused_i

    def _hops(self, acc, delta, idx):
        R = self.R
        two_n = 2 * R.n
        hops, fuse_ok = hop_split(delta, self.w, self.m if idx is not None else 0)
        fused_i = None
        for k, h in enumerate(hops):
            acc = R.automorphism(acc, pow(self.g, h, two_n))
            COUNTER.automorph += 1
            last = (k == len(hops) - 1)
            if last and fuse_ok and idx:
                fused_i = idx[0]
                acc = self.ctx.ext_prod(acc, self.fused[fused_i][h])
            else:
                acc = self.ctx.ext_prod(acc, self.ak[h])
        return acc, fused_i

    def _finish(self, acc, cur, L):
        R = self.R
        two_n = 2 * R.n
        if cur <= L - 1:
            d1 = (L - 1) - cur
            if d1 > 0:
                acc = self._hops(acc, d1, None)[0]
            acc = R.automorphism(acc, (-self.g) % two_n)
            COUNTER.automorph += 1
            acc = self.ctx.ext_prod(acc, self.ak_neg)
            if L - 1 > 0:
                acc = self._hops(acc, L - 1, None)[0]
        else:
            d2 = (2 * L - 1) - cur
            if d2 > 0:
                acc = self._hops(acc, d2, None)[0]
        return acc

    def ring_secret(self):
        return self.ctx.f_c
