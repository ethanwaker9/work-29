import numpy as np


def dlog_tables(two_n, g=5):
    L = two_n // 4
    sign = np.zeros(two_n, dtype=np.int64)
    ell = np.zeros(two_n, dtype=np.int64)
    cur = 1
    for e in range(L):
        sign[cur] = 1
        ell[cur] = e
        neg = (-cur) % two_n
        sign[neg] = -1
        ell[neg] = e
        cur = cur * g % two_n
    return sign, ell


def hop_cost(delta, w, m):
    if delta <= 0:
        return 0
    if delta <= m:
        return 0
    r0 = -(-delta // w)
    if delta - m <= (r0 - 1) * w:
        return r0 - 1
    return r0


def hop_split(delta, w, m):
    if delta <= m:
        return [delta], True
    r0 = -(-delta // w)
    if delta - m <= (r0 - 1) * w:
        r = r0
        last = m
        rest = delta - m
        hops = []
        while rest > 0:
            h = min(w, rest)
            hops.append(h)
            rest -= h
        hops.append(last)
        return hops, True
    hops = []
    rest = delta
    while rest > 0:
        h = min(w, rest)
        hops.append(h)
        rest -= h
    return hops, False


def build_stops(alpha_odd, two_n, g=5):
    sign, ell = dlog_tables(two_n, g)
    L = two_n // 4
    buckets = [[] for _ in range(2 * L)]
    for i, w_i in enumerate(alpha_odd):
        s = sign[w_i]
        e = ell[w_i]
        if s < 0:
            stop = (L - 1) - e
        else:
            stop = L + (L - 1) - e
        buckets[stop].append(i)
    return buckets, L


def round_to_odd(x):
    y = 2 * np.rint((x - 1) / 2.0) + 1
    return np.where(np.abs(x) < 1.0, 0.0, y)


def ms_alpha(a, b, q_ks, two_n, mode="plain"):
    sc = two_n / float(q_ks)
    x = -np.asarray(a, dtype=np.float64) * sc
    y = float(b) * sc
    if mode == "odd":
        al = round_to_odd(x).astype(np.int64)
    elif mode == "half":
        al = (2 * np.rint(x / 2.0)).astype(np.int64)
    else:
        al = np.rint(x).astype(np.int64)
    be = int(np.rint(y))
    return al % two_n, be % two_n


def build_stops_sub(alpha, active, two_n, g=5):
    sign, ell = dlog_tables(two_n, g)
    L = two_n // 4
    buckets = [[] for _ in range(2 * L)]
    for i in active:
        w_i = int(alpha[i])
        s = sign[w_i]
        e = ell[w_i]
        stop = (L - 1) - e if s < 0 else L + (L - 1) - e
        buckets[stop].append(i)
    return buckets, L
