G_CALLDATA_NONZERO = 16
G_CALLDATA_ZERO = 4
G_SSTORE_SET = 20000
G_SSTORE_RESET = 2900
G_SLOAD_WARM = 100
G_LOG_BYTE = 8
G_LOG_BASE = 375
WORD = 32


def ct_bytes(n, log_q):
    return (n + 1) * log_q // 8 + 1


def words(nbytes):
    return (nbytes + WORD - 1) // WORD


def calldata_gas(nbytes, zero_fraction=0.0):
    nz = int(nbytes * (1.0 - zero_fraction))
    z = nbytes - nz
    return nz * G_CALLDATA_NONZERO + z * G_CALLDATA_ZERO


def storage_gas(nbytes, fresh=False):
    w = words(nbytes)
    return w * (G_SSTORE_SET if fresh else G_SSTORE_RESET)


def handle_gas(fresh=False):
    return G_SSTORE_SET if fresh else G_SSTORE_RESET


def event_gas(nbytes, topics=2):
    return G_LOG_BASE * (1 + topics) + nbytes * G_LOG_BYTE


def transfer_cost(n, log_q, model="onchain", updated_slots=2):
    cb = ct_bytes(n, log_q)
    if model == "onchain":
        g = calldata_gas(cb) + updated_slots * storage_gas(cb)
        return dict(model=model, ct_bytes=cb, gas=g,
                    storage_words=updated_slots * words(cb))
    g = calldata_gas(cb) + updated_slots * handle_gas() + event_gas(cb)
    return dict(model=model, ct_bytes=cb, gas=g, storage_words=updated_slots)
