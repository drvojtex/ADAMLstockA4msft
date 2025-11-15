
import os
import pandas as pd
import numpy as np

from typing import List, Tuple, Union


class CSVDataLoader:
    """
    An interface for loading CSV data into a pandas DataFrame.

    This class provides methods for checking file existence, 
    lazy-loading data, and reloading when needed.

    Example:
        >>> loader = CSVDataLoader("data.csv")
        >>> df = loader.load()
        >>> print(df.head())
    """

    def __init__(self, filepath: str, **kwargs):
        """
        Initialize the CSVDataLoader.

        Args:
            filepath (str): Path to the CSV file.
            **kwargs: Additional keyword arguments passed to `pandas.read_csv`,
                such as `parse_dates`, `dtype`, `sep`, etc.
        """
        self.filepath = filepath
        self.kwargs = kwargs
        self._data = None

    def _check_file(self) -> None:
        """
        Check if the CSV file exists.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(f"File '{self.filepath}' not found.")

    def load(self) -> pd.DataFrame:
        """
        Load the CSV file into a pandas DataFrame.

        The data is read only once and cached for future access. 
        Use `reload()` if you want to refresh the data.

        Returns:
            pd.DataFrame: The loaded data.
        """
        if self._data is None:
            self._check_file()
            self._data = pd.read_csv(self.filepath, **self.kwargs)
        return self._data

    def reload(self) -> pd.DataFrame:
        """
        Force a reload of the CSV file, ignoring the cached data.

        Returns:
            pd.DataFrame: The newly loaded data.
        """
        self._check_file()
        self._data = pd.read_csv(self.filepath, **self.kwargs)
        return self._data

    @property
    def data(self) -> pd.DataFrame:
        """
        Return the already loaded data.

        Raises:
            ValueError: If data has not been loaded yet.

        Returns:
            pd.DataFrame: The cached pandas DataFrame.
        """
        if self._data is None:
            raise ValueError("Data not loaded yet. Call `.load()` first.")
        return self._data

    def split_by_year(
        self,
        train_years: List[int],
        test_years: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into X_train and X_test based on years.
        X contains: Open, High, Low, Volume, Close + one-hot date features 
        (day of week, week of month, month of year).
        """

        # Load data
        df = self.load()
        if 'Date' not in df.columns:
            raise ValueError("DataFrame must contain a 'Date' column.")
        df['Date'] = pd.to_datetime(df['Date'])

        # Add Year column
        df['Year'] = df['Date'].dt.year

        # Required columns
        base_cols = ['Open', 'High', 'Low', 'Volume', 'Close']
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        # Train/test splits
        df_train = df[df['Year'].isin(train_years)].sort_values('Date').reset_index(drop=True)
        df_test  = df[df['Year'].isin(test_years)].sort_values('Date').reset_index(drop=True)

        # Base feature matrices
        X_train = df_train[base_cols].values
        X_test  = df_test[base_cols].values

        # -------------------------------------------------------
        # Add one-hot encoded date features (22 columns)
        # -------------------------------------------------------
        def build_date_features(df_in):
            # Monday–Friday -> weekday 0–4
            dow = df_in['Date'].dt.weekday.clip(0, 4)

            # Week of month: 1–5
            wom = ((df_in['Date'].dt.day - 1) // 7) + 1

            # Month of year: 1–12
            moy = df_in['Date'].dt.month

            # one-hot encodings
            one_hot_dow = np.eye(5)[dow]          # (n, 5)
            one_hot_wom = np.eye(5)[wom - 1]      # (n, 5)
            one_hot_moy = np.eye(12)[moy - 1]     # (n, 12)

            # final concat: shape (n, 22)
            return np.hstack([one_hot_dow, one_hot_wom, one_hot_moy])

        # Build and append date features
        date_train = build_date_features(df_train)
        date_test  = build_date_features(df_test)

        X_train = np.hstack([X_train, date_train])
        X_test  = np.hstack([X_test, date_test])

        return X_train, X_test
