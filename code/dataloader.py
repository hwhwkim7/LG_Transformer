import numpy as np
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, windows_path, labels_path, sid_path):
        self.X = np.load(windows_path)  # (N, L, C)
        self.y = np.load(labels_path)  # (N,)
        self.sid = np.load(sid_path)   # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # shape (L, C)
        y = self.y[idx]  # 0 or 1
        sid = self.sid[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(sid, dtype=torch.float32)