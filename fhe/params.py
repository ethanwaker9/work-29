from dataclasses import dataclass


@dataclass
class Params:
    name: str = "P128"
    N: int = 1024
    logQ: int = 27
    d: int = 3
    base_bits: int = 8
    n: int = 512
    key_dist: str = "ternary"
    key_sigma: float = 3.2
    sigma: float = 3.2
    sigma_fresh: float = 819.2
    sigma_f: float = 32.0
    q_ks: int = 1 << 18
    ks_base_bits: int = 4
    d_ks: int = 5
    window: int = 16
    fuse: int = 1
    ap_base: int = 32
    lam: int = 128
    parties: int = 1


PT = Params(name="PT", key_dist="ternary", n=512, parties=1)
PG = Params(name="PG", key_dist="gaussian", key_sigma=1.1547, n=465, parties=1)
PTH = Params(name="PTH", key_dist="gaussian", key_sigma=2.3094, n=465, parties=8)

SETS = {"PT": PT, "PG": PG, "PTH": PTH}
METHODS = ["AP", "GINX", "LMKCDEY", "FINAL", "XZD", "Ours"]
