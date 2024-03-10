import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StringDataset(Dataset):
    def __init__(self, strings):
        self.strings = strings

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        return self.strings[idx]


def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
