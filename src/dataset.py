import torch
import numpy as np
from torch.utils.data import Dataset


class StockTorchDataset(Dataset):
    def __init__(self, data: np.ndarray, zscore_params=None, history=5, target_days=[1]):
        """
        data: numpy matrix shape (N, D)
              - col 4  = close price
              - col 5..k = next 22 feature one-hot encoding data columns

        zscore_params: tuple (mu, sigma) or None

        history: number of past days included in the sample

        target_days: list of integers, default [1]
              - Specifies which future days (relative to real_idx)
                will be predicted.
              - Example: [1, 5] → model predicts next-day and next-5-days change.
        """
        self.history = history
        self.target_days = sorted(target_days)

        # --- 1) Extract close column (column 4) ---
        close = data[:, 4]

        # --- 2) Compute overnight change ---
        change = np.zeros_like(close)
        change[1:] = close[1:] - close[:-1]

        # --- 3) Standardize ---
        if zscore_params is None:
            self.mu = change.mean()
            self.sigma = change.std() if change.std() > 0 else 1.0
        else:
            self.mu, self.sigma = zscore_params

        change_std = (change - self.mu) / self.sigma

        # --- 4) Other feature columns ---
        other_features = data[:, 5:]

        # --- 5) Final matrix X ---
        # column 0 = standardized change
        # columns 1..22 = other features
        self.X = np.column_stack([change_std, other_features]).astype(np.float32)

    def get_zscore_params(self):
        return self.mu, self.sigma

    def __len__(self):
        """
        valid sample index real_idx must allow:
        - accessing history_vec → needs real_idx - history >= 0
        - accessing the furthest target → needs real_idx + max(target_days) < len(X)
        """
        max_target = max(self.target_days)

        # Last valid real_idx is len(X) - 1 - max_target
        last_real_idx = self.X.shape[0] - 1 - max_target

        # First valid real_idx is history
        first_real_idx = self.history

        return max(0, last_real_idx - first_real_idx + 1)

    def __getitem__(self, idx):
        # map dataset index → real index in X
        real_idx = idx + self.history

        # ----- input vector -----
        history_vec = self.X[real_idx - self.history : real_idx, 0]
        today_change = self.X[real_idx, 0]
        today_features = self.X[real_idx, 1:]

        x = np.concatenate([history_vec, [today_change], today_features]).astype(np.float32)
        x = torch.from_numpy(x)

        # ----- targets -----
        y_values = []
        for t in self.target_days:
            y_values.append(self.X[real_idx + t, 0])  # standardized change at t days ahead

        y = torch.tensor(y_values, dtype=torch.float32)
        return x, y
