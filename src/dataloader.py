
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

    def split_by_year(self, train_years: List[int], test_years: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter data based on specified years for training and testing, 
        and split into features (X) and target (y) NumPy arrays.

        Features (X) include 'Open', 'High', 'Low', 'Volume'.
        Target (y) is 'Close'.
        The order of days is preserved from the raw dataframe.

        Args:
            train_years (List[int]): List of years to include in the training set.
            test_years (List[int]): List of years to include in the testing set.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (X_train, y_train, X_test, y_test)
        
        Raises:
            ValueError: If data has not been loaded or 'Date' column is missing/incorrectly formatted.
        """
        
        # Load and check data
        df = self.load()
        if 'Date' not in df.columns:
            raise ValueError("DataFrame must contain a 'Date' column.")
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check the Date column format
        try:
            df['Year'] = df['Date'].dt.year
        except AttributeError:
            raise ValueError("'Date' column is not in datetime format. Check `parse_dates` in initialization.")
            
        # Define and check features columns
        feature_cols = ['Open', 'High', 'Low', 'Volume']
        target_col = 'Close'
        missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in data: {', '.join(missing_cols)}")

        # Create training data matrix
        train_mask = df['Year'].isin(train_years)
        df_train = df[train_mask].sort_values(by='Date').reset_index(drop=True)
        
        # Create testing data matrix
        test_mask = df['Year'].isin(test_years)
        df_test = df[test_mask].sort_values(by='Date').reset_index(drop=True)

        # Split into input and target matrices
        X_train = df_train[feature_cols].values
        y_train = df_train[target_col].values
        X_test = df_test[feature_cols].values
        y_test = df_test[target_col].values
        
        return X_train, y_train, X_test, y_test
