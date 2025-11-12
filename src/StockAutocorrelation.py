import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import warnings


class StockAutocorrelation:
    def __init__(self, dpi = 80):
        self.dpi = dpi
        # Suppress pandas/Plotly FutureWarnings
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DatetimeProperties.to_pydatetime is deprecated",
            category=FutureWarning
        )

    def prepare_data(self, df: pd.DataFrame, data_name: str = "Close", idx_start: int = 0, idx_end: int = 50):
        """Loads and validates input data"""
        self.df = df
        self.data_name = data_name
        self.idx_start = idx_start
        self.idx_end = idx_end

        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        
        if not(self.data_name in required):
            raise ValueError(f"Invalid data name: {self.data_name}")
        
        self.data = df[self.data_name][self.idx_start:self.idx_end] # [50:100]

    def plot_autocorelation(self, lags: int = 20):
        """ Compute and render autocorelation of 'data_name'. """
        fig, ax = plt.subplots(figsize=(7,4), dpi=self.dpi)
        plot_acf(self.data, lags=lags, ax=ax)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Autocorrelation of {self.data_name}")
        plt.show()
    
    def plot_lag(self):
        """ Plot lag """
        fig2, ax2 = plt.subplots(figsize=(6,5), dpi=self.dpi)
        pd.plotting.lag_plot(self.data, lag=1, ax=ax2)
        ax2.set_xlabel("y(t)")
        ax2.set_ylabel("y(t+1)")
        ax2.set_title(f"Lag plot of {self.data_name}")
        plt.show()
    
    def correlation_across_multiple_lags(self):
        # Create lagged dataset
        df = pd.concat([self.data.shift(3), self.data.shift(2),
                        self.data.shift(1), self.data], axis=1)

        # Set column names
        df.columns = ['t', 't+1', 't+2', 't+3']

        res = df.corr()
        print(res)
