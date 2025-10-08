import os, random, numpy as np

def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed)
    return seed
