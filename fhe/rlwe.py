import numpy as np
from .ring import sample_gaussian
from .counters import COUNTER


class RlweCtx:
    def __init__(self, ring, gadget, sigma, rng):
        self.R = ring
        self.G = gadget
        self.sigma = sigma
        self.rng = rng
        self.Q = ring.q
        self.N = ring.n

    def keygen(self, key_dist="ternary"):
        from .ring import sample_secret
        zc = sample_secret(self.rng, key_dist, self.N)
        self.z_c = zc
        self.z = zc % self.Q
        self.z_hat = self.R.ntt(self.z)
        return self.z

    def encrypt_rlwe(self, m, z=None):
        z_hat = self.z_hat if z is None else self.R.ntt(z)
        a = self.rng.integers(0, self.Q, self.N, dtype=np.int64)
        e = sample_gaussian(self.rng, self.sigma, self.N)
        b = (self.R.intt(self.R.ntt(a) * z_hat % self.Q) + e + m) % self.Q
        return np.stack([a, b])

    def phase(self, ct, z=None):
        z_hat = self.z_hat if z is None else self.R.ntt(z)
        az = self.R.intt(self.R.ntt(ct[0]) * z_hat % self.Q)
        return self.R.center(ct[1] - az)

    def rlwe_prime(self, m, z=None):
        out = np.empty((self.G.d, 2, self.N), dtype=np.int64)
        for j in range(self.G.d):
            out[j] = self.encrypt_rlwe(m * self.G.powers[j] % self.Q, z)
        return out

    def rgsw(self, m, z=None):
        zz = self.z if z is None else z % self.Q
        mz = self.R.mul(m % self.Q, zz)
        return np.stack([self.rlwe_prime((-mz) % self.Q, z), self.rlwe_prime(m % self.Q, z)])

    def prep_rgsw(self, C):
        return self.R.ntt(C).astype(np.int32)

    def prep_rlwep(self, C):
        return self.R.ntt(C).astype(np.int32)

    def ext_prod(self, ct, Chat):
        R = self.R
        Q = self.Q
        d = self.G.d
        dec = self.G.decompose(ct)
        dh = R.ntt(np.ascontiguousarray(dec.transpose(1, 0, 2)))
        COUNTER.ntt += 2 * d
        acc = (dh[0][:, None, :] * Chat[0]).sum(axis=0) % Q
        acc = (acc + (dh[1][:, None, :] * Chat[1]).sum(axis=0)) % Q
        COUNTER.ringmul += 4 * d
        COUNTER.extprod += 1
        out = R.intt(acc)
        COUNTER.intt += 2
        return out

    def keyswitch_prime(self, ct_a, ct_b, Khat):
        R = self.R
        Q = self.Q
        dec = self.G.decompose(ct_a)
        dh = R.ntt(dec)
        COUNTER.ntt += self.G.d
        acc = (dh[:, None, :] * Khat).sum(axis=0) % Q
        COUNTER.ringmul += 2 * self.G.d
        out = R.intt(acc)
        COUNTER.intt += 2
        out[1] = (out[1] + ct_b) % Q
        COUNTER.keyswitch += 1
        return out

    def auto_key(self, t, z=None):
        zz = self.z if z is None else z
        zt = self.R.automorphism(zz, t)
        return self.rlwe_prime((-zt) % self.Q, zz)

    def apply_auto(self, ct, t, Khat):
        at = self.R.automorphism(ct[0], t)
        bt = self.R.automorphism(ct[1], t)
        COUNTER.automorph += 1
        return self.keyswitch_prime(at, bt, Khat)
