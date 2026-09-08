import time


class GateCounter:
    def __init__(self):
        self.count = 0
        self.free = 0
        self.time = 0.0


class Evaluator:
    def __init__(self, scheme, counter=None, pre=None):
        self.s = scheme
        self.c = counter or GateCounter()
        self.pre = pre

    def _boot(self, a, b):
        self.c.count += 1
        if self.pre is not None:
            self.pre(a, b)
        t0 = time.perf_counter()
        out = self.s.bootstrap(a, b)
        self.c.time += time.perf_counter() - t0
        return out

    def _lin(self, coeffs, cts, const):
        q = self.s.p.q_ks
        import numpy as np
        a = np.zeros(self.s.p.n, dtype=np.int64)
        bb = const % q
        for k, ct in zip(coeffs, cts):
            a = (a + k * np.asarray(ct[0])) % q
            bb = (bb + k * ct[1]) % q
        return a % q, bb % q

    def NOT(self, x):
        import numpy as np
        q = self.s.p.q_ks
        self.c.free += 1
        return (-np.asarray(x[0])) % q, (q // 4 - x[1]) % q

    def NAND(self, x, y):
        q = self.s.p.q_ks
        return self._boot(*self._lin([-1, -1], [x, y], 5 * q // 8))

    def AND(self, x, y):
        q = self.s.p.q_ks
        return self._boot(*self._lin([1, 1], [x, y], -(q // 8)))

    def OR(self, x, y):
        q = self.s.p.q_ks
        return self._boot(*self._lin([1, 1], [x, y], q // 8))

    def NOR(self, x, y):
        q = self.s.p.q_ks
        return self._boot(*self._lin([-1, -1], [x, y], 3 * q // 8))

    def XOR(self, x, y):
        return self.AND(self.OR(x, y), self.NAND(x, y))

    def XNOR(self, x, y):
        return self.NOT(self.XOR(x, y))

    def MUX(self, sel, x, y):
        return self.OR(self.AND(sel, x), self.AND(self.NOT(sel), y))


def full_adder(ev, a, b, cin):
    x = ev.XOR(a, b)
    s = ev.XOR(x, cin)
    cout = ev.OR(ev.AND(a, b), ev.AND(x, cin))
    return s, cout


def ripple_add(ev, A, B, cin=None):
    out = []
    c = cin
    for i in range(len(A)):
        if c is None:
            s = ev.XOR(A[i], B[i])
            c = ev.AND(A[i], B[i])
        else:
            s, c = full_adder(ev, A[i], B[i], c)
        out.append(s)
    return out, c


def ripple_sub(ev, A, B):
    nb = [ev.NOT(x) for x in B]
    out = []
    c = None
    for i in range(len(A)):
        if i == 0:
            s = ev.XNOR(A[0], nb[0])
            c = ev.OR(A[0], nb[0])
        else:
            s, c = full_adder(ev, A[i], nb[i], c)
        out.append(s)
    return out, c


def geq(ev, A, B):
    r = None
    for i in range(len(A)):
        eq = ev.XNOR(A[i], B[i])
        gt = ev.AND(A[i], ev.NOT(B[i]))
        if r is None:
            r = ev.OR(gt, eq)
        else:
            r = ev.OR(gt, ev.AND(eq, r))
    return r


def mux_word(ev, sel, A, B):
    return [ev.MUX(sel, a, b) for a, b in zip(A, B)]


def zero_word(ev, width, enc_zero):
    return [enc_zero() for _ in range(width)]
