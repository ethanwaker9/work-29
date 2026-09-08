import numpy as np
from .ring import sample_gaussian
from .counters import COUNTER


class NgsCtx:
    def __init__(self, ring, gadget, sigma_g, rng, sigma_f=None):
        self.R = ring
        self.G = gadget
        self.sigma_g = sigma_g
        self.sigma_f = sigma_g if sigma_f is None else sigma_f
        self.rng = rng
        self.Q = ring.q
        self.N = ring.n

    def keygen(self):
        while True:
            f = sample_gaussian(self.rng, self.sigma_f, self.N)
            f[0] += 1
            finv = self.R.inverse(f % self.Q)
            if finv is not None:
                self.f = f % self.Q
                self.f_c = f
                self.finv = finv
                self.f_hat = self.R.ntt(self.f)
                return self.f

    def vector_ct(self, v, out_key_inv=None):
        Q = self.Q
        ki = self.finv if out_key_inv is None else out_key_inv
        d = self.G.d
        g = sample_gaussian(self.rng, self.sigma_g, (d, self.N)) % Q
        gk = self.R.intt(self.R.ntt(g) * self.R.ntt(ki)[None, :] % Q)
        return (gk + self.G.powers[:, None] * (v % Q)) % Q

    def prep(self, C):
        return self.R.ntt(C).astype(np.int32)

    def ext_prod(self, c, Chat):
        Q = self.Q
        dec = self.G.decompose(c)
        dh = self.R.ntt(dec)
        COUNTER.ntt += self.G.d
        acc = (dh * Chat).sum(axis=0) % Q
        COUNTER.ringmul += self.G.d
        COUNTER.extprod += 1
        out = self.R.intt(acc)
        COUNTER.intt += 1
        return out

    def phase(self, c):
        return self.R.center(self.R.intt(self.R.ntt(c) * self.f_hat % self.Q))
