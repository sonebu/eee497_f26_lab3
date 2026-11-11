import numpy as np

fs = 2_048_000


def load_iq(path):
    raw = np.fromfile(path, dtype=np.uint8).astype(np.float64) - 127.0
    x = raw[0::2] + 1j * raw[1::2]
    return x / 128.0, float(fs)
