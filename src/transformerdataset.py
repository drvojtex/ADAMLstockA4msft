
import torch
import numpy as np

from torch.utils.data import Dataset


class StockTorchTransDataset(Dataset):
    def __init__(self, data: np.ndarray, zscore_params=None, history=5, target_days=[1]):
        """
        data: numpy matrix shape (N, D)
              - cols 1..5 = price-type columns (we use all five)
              - cols 5..k = other feature columns

        We compute:
            diff on columns 1..4
            zscore on columns 1..5
            and then append other_features
        """
        self.history = history
        self.target_days = sorted(target_days)

        # --- 1) Extract columns 0-4 ---
        base = data[:, 0:5].astype(np.float32)     # shape (N, 5)
        # print(base)

        # --- 2) Diference columns 1..4 ---
        diff = np.zeros_like(base)
        diff[1:, :3] = base[1:, :3] - base[:-1, :3]
        diff[:, 3] = base[:, 3]                    # Volume stays unchanged before zscore
        diff[1:, 4] = base[1:, 4] - base[:-1, 4]   # Close value

        # --- 3) Z-score (per-column) ---
        if zscore_params is None:
            self.mu = diff.mean(axis=0)
            self.sigma = diff.std(axis=0)
            self.sigma[self.sigma == 0] = 1.0
        else:
            self.mu, self.sigma = zscore_params

        diff_std = (diff - self.mu) / self.sigma   # shape (N, 5)

        # --- 4) Other features ---
        other_features = data[:, 5:].astype(np.float32)  # shape (N, M)

        # --- 5) Final matrix X ---
        # Concatenate standardized 5 columns + other_features
        self.X = np.concatenate([diff_std, other_features], axis=1)  # shape (N, 5+M)

        # Save number of columns
        self.F_all = self.X.shape[1]   # 5+M
        self.F_clean = 5               # only 1..5 for target

    def get_zscore_params(self):
        return self.mu, self.sigma

    def __len__(self):
        max_target = max(self.target_days)
        last_real_idx = self.X.shape[0] - 1 - max_target
        first_real_idx = self.history
        return max(0, last_real_idx - first_real_idx + 1)

    def __getitem__(self, idx):
        real_idx = idx + self.history

        # ----- Input X: matrix (history+1 rows, F_all cols) -----
        # rows = [real_idx-history , ..., real_idx]
        x_mat = self.X[real_idx - self.history : real_idx + 1]  # shape (history+1, F_all)
        x = torch.tensor(x_mat, dtype=torch.float32)

        # ----- Targets: preprocessed 5-column data only -----
        y_list = []
        for t in self.target_days:
            y_list.append(self.X[real_idx + t, :self.F_clean])   # first 5 columns only

        y = torch.tensor(np.stack(y_list), dtype=torch.float32)  # shape (T, 5)

        return x, y
