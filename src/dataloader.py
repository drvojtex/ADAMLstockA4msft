
import os
import pandas as pd


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
