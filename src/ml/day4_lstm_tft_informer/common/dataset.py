"""
dataset.py
DL 시퀀스 학습용 torch Dataset. SequencePreprocessor.transform() 결과를 텐서로 감싼다.
"""

import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, transformed: dict):
        self.time_varying = torch.as_tensor(transformed["time_varying"], dtype=torch.float32)
        self.static_cont = torch.as_tensor(transformed["static_cont"], dtype=torch.float32)
        self.static_cat = torch.as_tensor(transformed["static_cat"], dtype=torch.long)
        self.target = torch.as_tensor(transformed["target"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> dict:
        return {
            "time_varying": self.time_varying[idx],
            "static_cont": self.static_cont[idx],
            "static_cat": self.static_cat[idx],
            "target": self.target[idx],
        }
