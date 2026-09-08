import numpy as np
from .ring import Ring, Gadget, find_ntt_prime, sample_secret, sample_gaussian
from .lwe import LweKeySwitcher, mod_switch, extract_rlwe, extract_ngs, lwe_phase, center
from .blindrot import (APRotator, GINXRotator, LMKRotator, NgsGinxRotator,
                       XZDRotator, FusedRotator, gadget_u_set)
from .counters import COUNTER


def nand_testvector(N, Q):
    tv = np.empty(N, dtype=np.int64)
    half = N // 2
    tv[:half + 1] = (-(Q // 8)) % Q
    tv[half + 1:] = Q // 8
    return tv


class FHEWScheme:
    def __init__(self, params, method, rng=None, sigma_f=None):
        p = params
        self.p = p
        self.rng = np.random.default_rng(20240915) if rng is None else rng
        self.Q = find_ntt_prime(p.N, p.logQ)
        self.R = Ring(p.N, self.Q)
        self.G = Gadget(self.R, p.base_bits, p.d)
        self.method = method
        if p.key_dist == "gaussian":
            bound = int(np.ceil(4.0 * p.key_sigma))
            while True:
                self.s = sample_secret(self.rng, p.key_dist, p.n, p.key_sigma)
                self.s = np.clip(self.s, -bound, bound)
                break
        else:
            self.s = sample_secret(self.rng, p.key_dist, p.n, p.key_sigma)
            bound = int(max(1, np.max(np.abs(self.s))))
        self.bound = bound
        self.uset = gadget_u_set(p.key_dist if p.key_dist in ("binary", "ternary") else "g", bound)
        sf = p.sigma_f if sigma_f is None else sigma_f
        if method == "AP":
            self.rot = APRotator(self.R, self.G, p.sigma, self.rng, p.n, p.q_ks, p.ap_base)
        elif method == "GINX":
            self.rot = GINXRotator(self.R, self.G, p.sigma, self.rng, p.n, p.q_ks, self.uset)
        elif method == "LMKCDEY":
            self.rot = LMKRotator(self.R, self.G, p.sigma, self.rng, p.n, p.q_ks, window=p.window)
        elif method == "FINAL":
            self.rot = NgsGinxRotator(self.R, self.G, p.sigma, self.rng, p.n, p.q_ks, self.uset, sf)
        elif method == "XZD":
            self.rot = XZDRotator(self.R, self.G, p.sigma, self.rng, p.n, p.q_ks, sf)
        elif method == "Ours":
            self.rot = FusedRotator(self.R, self.G, p.sigma, self.rng, p.n, p.q_ks,
                                    window=p.window, fuse=p.fuse, sigma_f=sf)
        else:
            raise ValueError(method)
        self.is_rlwe = method in ("AP", "GINX", "LMKCDEY")
        self.tv = nand_testvector(p.N, self.Q)

    def keygen(self):
        self.rot.keygen(self.s)
        z = self.rot.ring_secret()
        self.ksw = LweKeySwitcher(self.rng, self.p.q_ks, self.p.ks_base_bits,
                                  self.p.d_ks, self.p.sigma).keygen(z, self.s)
        return self

    def key_bytes(self):
        logq = int(np.ceil(np.log2(self.Q)))
        br = self.rot.key_elements() * self.p.N * logq / 8.0
        logks = int(np.ceil(np.log2(self.p.q_ks)))
        ks = self.ksw.elements() * logks / 8.0
        return br, ks

    def encrypt(self, m):
        q = self.p.q_ks
        a = self.rng.integers(0, q, self.p.n, dtype=np.int64)
        e = int(sample_gaussian(self.rng, self.p.sigma_fresh, 1)[0])
        b = (int(np.dot(a, self.s)) + q // 4 * int(m) + e) % q
        return a, b

    def decrypt(self, ct):
        q = self.p.q_ks
        ph = lwe_phase(ct[0], ct[1], self.s, q)
        return int(np.rint(4.0 * ph / q)) % 4

    def noise(self, ct, m):
        q = self.p.q_ks
        ph = lwe_phase(ct[0], ct[1], self.s, q)
        return center(ph - q // 4 * int(m), q)

    def nand(self, ct1, ct2):
        q = self.p.q_ks
        a = (-(np.asarray(ct1[0]) + np.asarray(ct2[0]))) % q
        b = (5 * q // 8 - ct1[1] - ct2[1]) % q
        return self.bootstrap(a, b)

    def bootstrap(self, a, b):
        acc = self.rot.rotate(a, b, self.p.q_ks, self.tv)
        if self.is_rlwe:
            ea, eb = extract_rlwe(acc, self.Q)
        else:
            ea, eb = extract_ngs(acc, self.Q)
        eb = (eb + self.Q // 8) % self.Q
        ka, kb = mod_switch((ea, eb), self.Q, self.p.q_ks)
        return self.ksw.switch(ka, kb)
