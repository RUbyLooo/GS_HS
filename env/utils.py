import numpy as np
from typing import Optional, Tuple

class SeedError(Exception):
    pass

def np_random(seed: Optional[int] = None) -> Tuple[np.random.Generator, int]:
    """
    Initialize and return a NumPy random Generator and the seed used.

    Args:
        seed: Optional integer seed. If None, a random seed is used.

    Returns:
        A tuple of (np.random.Generator, seed used as int)
    """
    if seed is not None:
        if not isinstance(seed, int):
            raise SeedError(f"Seed must be a python int, got: {type(seed)}")
        if seed < 0:
            raise SeedError(f"Seed must be non-negative, got: {seed}")

    seed_seq = np.random.SeedSequence(seed)
    generator = np.random.default_rng(seed_seq)
    return generator, seed_seq.entropy
