import os, random
import numpy
def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)  # fixes hash randomization
    random.seed(seed)                          # Python built-in RNG
    numpy.random.seed(seed)                       # NumPy RNG