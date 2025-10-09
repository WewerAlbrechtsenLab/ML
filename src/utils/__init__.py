import os, random, numpy as np

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)  # fixes hash randomization
    random.seed(seed)                          # Python built-in RNG
    np.random.seed(seed)                       # NumPy RNG