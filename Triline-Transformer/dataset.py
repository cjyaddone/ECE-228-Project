from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TrilineWindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        log_distance_targets: np.ndarray,
        direction_targets: np.ndarray,
        bird_ids: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.features = torch.as_tensor(features[indices], dtype=torch.float32)
        self.labels = torch.as_tensor(labels[indices], dtype=torch.float32)
        self.log_distance_targets = torch.as_tensor(
            log_distance_targets[indices], dtype=torch.float32
        )
        self.direction_targets = torch.as_tensor(direction_targets[indices], dtype=torch.float32)
        self.bird_ids = torch.as_tensor(bird_ids[indices], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.features[idx],
            self.bird_ids[idx],
            self.labels[idx],
            self.log_distance_targets[idx],
            self.direction_targets[idx],
        )


FlyNoflyDataset = TrilineWindowDataset
