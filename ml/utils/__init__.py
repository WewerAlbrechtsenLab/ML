import os, random, torch, numpy as np

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)  # fixes hash randomization
    random.seed(seed)                          # Python built-in RNG
    np.random.seed(seed)                       # NumPy RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False