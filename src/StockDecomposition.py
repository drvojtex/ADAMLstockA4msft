import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


class StockDecomposition:
	"""
	Class for time-series decomposition of stock data (trend, seasonality, residual).
	DataFrame loads from CSV through `set_data()`.
	"""
	def __init__(self, period: int = 252):
		self.period = period
		self.ts = None
		self.result = None

	def set_data(self, df: pd.DataFrame):
		"""
		Set the data from a DataFrame and validate required columns.
		"""
		required = {"Date", "Open", "High", "Low", "Close", "Volume"}
		if not required.issubset(df.columns):
			missing = required - set(df.columns)
			raise ValueError(f"DataFrame is missing required columns: {missing}")
		self.df = df.sort_values('Date')
		self.ts = self.df.set_index('Date')['Close']
		self.ts.name = ""  # Remove 'Close' title from decomposition subplots

	def decompose(self):
		"""
		Perform seasonal decomposition (trend, seasonal, residual).
		"""
		self.result = seasonal_decompose(self.ts, model='multiplicative', period=self.period)

	def plot(self, out_path: str = None):
		"""
		Plot or save the decomposition results with high quality.
		"""
		fig = self.result.plot()
		fig.set_size_inches((12, 8))
		# Make residual dots smaller
		axes = fig.get_axes()
		if len(axes) >= 4:
			resid_ax = axes[3]
			for line in resid_ax.lines:
				if hasattr(line, 'get_marker') and line.get_marker() == 'o':
					line.set_markersize(2)  # Set smaller marker size
		plt.suptitle('Time-Series Decomposition of MSFT Stock Closing Prices', fontsize=16)
		plt.tight_layout()
		if out_path:
			plt.savefig(out_path, dpi=300, bbox_inches='tight')
		else:
			plt.show()