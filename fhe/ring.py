import numpy as np


def _bit_reverse(i, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def find_ntt_prime(n, bits, headroom=0.99):
    m = 2 * n
    cap = int((1 << bits) * headroom)
    k = cap // m
    while k > 1:
        q = k * m + 1
        if q <= cap and _is_prime(q):
            return q
        k -= 1
    raise ValueError("no prime found")


def _is_prime(x):
    if x < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if x % p == 0:
            return x == p
    d = x - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        v = pow(a, d, x)
        if v == 1 or v == x - 1:
            continue
        for _ in range(r - 1):
            v = v * v % x
            if v == x - 1:
                break
        else:
            return False
    return True


def _primitive_root(q):
    fac = []
    x = q - 1
    d = 2
    while d * d <= x:
        if x % d == 0:
            fac.append(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        fac.append(x)
    g = 2
    while True:
        if all(pow(g, (q - 1) // f, q) != 1 for f in fac):
            return g
        g += 1


class Ring:
    def __init__(self, n, q):
        assert n & (n - 1) == 0
        assert (q - 1) % (2 * n) == 0
        self.n = n
        self.q = q
        self.logn = int(np.log2(n))
        g = _primitive_root(q)
        psi = pow(g, (q - 1) // (2 * n), q)
        psi_inv = pow(psi, q - 2, q)
        rev = [_bit_reverse(i, self.logn) for i in range(n)]
        self.psi_rev = np.array([pow(psi, rev[i], q) for i in range(n)], dtype=np.int64)
        self.psi_inv_rev = np.array([pow(psi_inv, rev[i], q) for i in range(n)], dtype=np.int64)
        self.n_inv = pow(n, q - 2, q)

    def ntt(self, a):
        a = np.asarray(a, dtype=np.int64).copy()
        shp = a.shape
        a = a.reshape(-1, self.n)
        q = self.q
        m = 1
        t = self.n
        while m < self.n:
            t >>= 1
            v = a.reshape(-1, m, 2, t)
            s = self.psi_rev[m:2 * m].reshape(1, m, 1)
            hi = v[:, :, 1, :] * s % q
            lo = v[:, :, 0, :].copy()
            v[:, :, 1, :] = (lo - hi) % q
            v[:, :, 0, :] = (lo + hi) % q
            a = v.reshape(-1, self.n)
            m <<= 1
        return a.reshape(shp)

    def intt(self, a):
        a = np.asarray(a, dtype=np.int64).copy()
        shp = a.shape
        a = a.reshape(-1, self.n)
        q = self.q
        t = 1
        h = self.n >> 1
        while h >= 1:
            v = a.reshape(-1, h, 2, t)
            s = self.psi_inv_rev[h:2 * h].reshape(1, h, 1)
            lo = v[:, :, 0, :].copy()
            hi = v[:, :, 1, :].copy()
            v[:, :, 0, :] = (lo + hi) % q
            v[:, :, 1, :] = (lo - hi) % q * s % q
            a = v.reshape(-1, self.n)
            t <<= 1
            h >>= 1
        a = a * self.n_inv % q
        return a.reshape(shp)

    def mul(self, a, b):
        return self.intt(self.ntt(a) * self.ntt(b) % self.q)

    def center(self, a):
        a = np.asarray(a, dtype=np.int64) % self.q
        return np.where(a > self.q // 2, a - self.q, a)

    def automorphism(self, a, t):
        n = self.n
        a = np.asarray(a, dtype=np.int64)
        idx = (np.arange(n) * t) % (2 * n)
        sign = np.where(idx >= n, -1, 1)
        pos = idx % n
        out = np.zeros(a.shape, dtype=np.int64)
        flat = a.reshape(-1, n)
        res = out.reshape(-1, n)
        contrib = flat * sign
        np.add.at(res, (slice(None), pos), contrib)
        return out % self.q

    def inverse(self, a):
        av = self.ntt(a)
        if np.any(av % self.q == 0):
            return None
        inv = np.array([pow(int(x), self.q - 2, self.q) for x in av], dtype=np.int64)
        return self.intt(inv)


class Gadget:
    def __init__(self, ring, base_bits, d):
        self.ring = ring
        self.base_bits = base_bits
        self.b = 1 << base_bits
        self.d = d
        q = ring.q
        logq = int(np.ceil(np.log2(q)))
        self.shift = max(0, logq - base_bits * d)
        self.powers = np.array([(1 << (self.shift + base_bits * j)) % q for j in range(d)], dtype=np.int64)

    def decompose(self, a):
        q = self.ring.q
        x = np.asarray(a, dtype=np.int64) % q
        x = np.where(x > q // 2, x - q, x)
        if self.shift:
            x = (x + (1 << (self.shift - 1))) >> self.shift
        out = np.empty((self.d,) + x.shape, dtype=np.int64)
        mask = self.b - 1
        half = self.b >> 1
        carry = np.zeros(x.shape, dtype=np.int64)
        for j in range(self.d):
            digit = ((x >> (self.base_bits * j)) & mask) + carry
            carry = (digit >= half).astype(np.int64)
            out[j] = digit - (carry << self.base_bits)
        return out


def sample_gaussian(rng, sigma, shape):
    return np.rint(rng.normal(0.0, sigma, size=shape)).astype(np.int64)


def sample_uniform(rng, q, shape):
    return rng.integers(0, q, size=shape, dtype=np.int64)


def sample_ternary(rng, shape):
    return rng.integers(-1, 2, size=shape, dtype=np.int64)


def sample_binary(rng, shape):
    return rng.integers(0, 2, size=shape, dtype=np.int64)


def sample_secret(rng, dist, shape, sigma=3.2):
    if dist == "binary":
        return sample_binary(rng, shape)
    if dist == "ternary":
        return sample_ternary(rng, shape)
    if dist == "gaussian":
        return sample_gaussian(rng, sigma, shape)
    raise ValueError(dist)
