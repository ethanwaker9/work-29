import numpy as np
from .ring import sample_gaussian


def mod_switch(ct, q_from, q_to):
    a = np.rint(np.asarray(ct[0], dtype=np.float64) * q_to / q_from).astype(np.int64) % q_to
    b = int(np.rint(float(ct[1]) * q_to / q_from)) % q_to
    return a, b


def lwe_phase(a, b, s, q):
    return int((b - int(np.dot(a % q, s))) % q)


def center(x, q):
    x = int(x) % q
    return x - q if x > q // 2 else x


def extract_rlwe(ct, Q):
    a, b = ct[0], ct[1]
    N = len(a)
    out = np.empty(N, dtype=np.int64)
    out[0] = a[0]
    out[1:] = (-a[1:][::-1]) % Q
    return out % Q, int(b[0]) % Q


def extract_ngs(c, Q):
    N = len(c)
    out = np.empty(N, dtype=np.int64)
    out[0] = c[0]
    out[1:] = (-c[1:][::-1]) % Q
    return (-out) % Q, 0


class LweKeySwitcher:
    def __init__(self, rng, q_ks, base_bits, d_ks, sigma):
        self.rng = rng
        self.q = q_ks
        self.bb = base_bits
        self.B = 1 << base_bits
        self.d = d_ks
        self.sigma = sigma

    def keygen(self, z, s):
        N = len(z)
        n = len(s)
        B = self.B
        d = self.d
        q = self.q
        cnt = N * d * (B - 1)
        a = self.rng.integers(0, q, size=(cnt, n), dtype=np.int64)
        e = sample_gaussian(self.rng, self.sigma, cnt)
        msg = np.empty(cnt, dtype=np.int64)
        k = 0
        zz = np.asarray(z, dtype=np.int64)
        for i in range(N):
            for j in range(d):
                for v in range(1, B):
                    msg[k] = v * (B ** j) * int(zz[i])
                    k += 1
        b = (a.dot(np.asarray(s, dtype=np.int64)) + e + msg) % q
        self.a = a.astype(np.int32)
        self.b = b.astype(np.int64)
        self.N = N
        self.n = n
        return self

    def elements(self):
        return self.N * self.d * (self.B - 1) * (self.n + 1)

    def switch(self, a, b):
        q = self.q
        B = self.B
        d = self.d
        N = self.N
        av = np.asarray(a, dtype=np.int64) % q
        digits = np.empty((N, d), dtype=np.int64)
        x = av.copy()
        for j in range(d):
            digits[:, j] = x % B
            x //= B
        rows = []
        base = (np.arange(N) * d * (B - 1))[:, None] + (np.arange(d) * (B - 1))[None, :]
        idx = base + digits - 1
        mask = digits > 0
        sel = idx[mask]
        acc_a = (-self.a[sel].astype(np.int64).sum(axis=0)) % q
        acc_b = (int(b) - int(self.b[sel].sum())) % q
        return acc_a, acc_b
