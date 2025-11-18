
import torch
import numpy as np
from torch.utils.data import Dataset


class StockTorchDataset(Dataset):
    def __init__(self, data: np.ndarray, zscore_params=None, history=5):
        """
        data: numpy matrix shape (N, D)
              - col 4  = close price
              - col 5..k = next 22 feature one-hot encoding data columns
        zscore_params: tuple (mu, sigma) or None
        history: int (default 5)
              - how many (previous) days should be included in the sample
        """
        self.history = history

        # --- 1) Extract close column (column 4) ---
        close = data[:, 4]

        # --- 2) Compute overnight change ---
        # change[0] = 0
        # change[i] = close[i] - close[i-1]
        change = np.zeros_like(close)
        change[1:] = close[1:] - close[:-1]

        # --- 3) Standardize ---
        if zscore_params is None:
            mu = change.mean()
            sigma = change.std() if change.std() > 0 else 1.0
            self.mu = mu
            self.sigma = sigma
        else:
            self.mu, self.sigma = zscore_params

        change_std = (change - self.mu) / self.sigma

        # --- 4) Add remaining feature columns (5..end) ---
        other_features = data[:, 5:]   # shape (N, 22)

        # --- 5) Final matrix (N, 23): standardized_change + 22 original cols ---
        self.X = np.column_stack([change_std, other_features]).astype(np.float32)

    def get_zscore_params(self):
        """
        Returns: tuple (mu, sigma)
              - zscore params
        """
        return self.mu, self.sigma

    def __len__(self):
        # Number of available samples:
        # The target is change_std at index (idx + 1),
        # so the latest valid idx is (N - history - 2).
        # Total samples = N - history - 1.
        return self.X.shape[0] - self.history - 1

    def __getitem__(self, idx):
        # Shift idx by history, because the first sample that
        # has enough past days starts at index = history.
        real_idx = idx + self.history

        # ----- input -----
        # Historical sequence of standardized changes
        history_vec = self.X[real_idx - self.history : real_idx, 0]   # shape (history,)

        # Today's standardized change
        today_change = self.X[real_idx, 0]

        # Today's additional 22 feature columns
        today_features = self.X[real_idx, 1:]                         # shape (22,)

        # Final input vector: [history changes | today's change | today's features]
        x = np.concatenate([history_vec, [today_change], today_features]).astype(np.float32)

        # ----- target -----
        # Target is tomorrow's standardized change
        y = self.X[real_idx + 1, 0].astype(np.float32)

        # Convert to torch tensors
        x = torch.from_numpy(x)
        y = torch.tensor(y)   # scalar tensor

        return x, y