import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import warnings


class StockVisualizer:
    """
    StockVisualizer
    ----------------
    Interactive 5D stock visualization using Plotly.

    The visualization displays:
      • Vertical bars representing the Low-High range (color-coded by Volume)
      • A dashed line representing Open prices
      • A solid line with markers representing Close prices
      • A colorbar showing Volume scale (logarithmic)

    Expected DataFrame columns:
        ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Name']
    """

    def __init__(
        self,
        colorscale: str = "Viridis",
        line_width: int = 4,
        open_line_color: str = "gray",
        close_line_color: str = "black",
        open_line_style: str = "dash",
        show_colorbar: bool = True,
        title: str = "Stock Visualization",
        template: str = "plotly_white",
        log_scale: bool = True
    ):
        # Store configuration parameters
        self.colorscale = colorscale
        self.line_width = line_width
        self.open_line_color = open_line_color
        self.close_line_color = close_line_color
        self.open_line_style = open_line_style
        self.show_colorbar = show_colorbar
        self.title = title
        self.template = template
        self.log_scale = log_scale

        # Suppress pandas/Plotly FutureWarnings
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DatetimeProperties.to_pydatetime is deprecated",
            category=FutureWarning
        )

    def visualize(self, df: pd.DataFrame) -> None:
        """Render interactive 5D stock visualization."""
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        df = df.copy()
        df["Date"] = np.array(pd.to_datetime(df["Date"]).dt.to_pydatetime())

        # Apply log scaling to Volume if requested
        if self.log_scale:
            vol_scaled = np.log1p(df["Volume"])
        else:
            vol_scaled = df["Volume"]

        # Normalize for color mapping
        normed_vol = (vol_scaled - vol_scaled.min()) / (vol_scaled.max() - vol_scaled.min())
        colors = sample_colorscale(self.colorscale, normed_vol)

        fig = go.Figure()

        # Add vertical bars (Low–High)
        for i in range(len(df)):
            fig.add_trace(go.Scatter(
                x=[df["Date"][i], df["Date"][i]],
                y=[df["Low"][i], df["High"][i]],
                mode="lines",
                line=dict(color=colors[i], width=self.line_width),
                showlegend=False
            ))

        # Add dashed line for Open
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Open"],
            mode="lines",
            line=dict(dash=self.open_line_style, color=self.open_line_color, width=2),
            name="Open"
        ))

        # Add solid line for Close
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines+markers",
            line=dict(color=self.close_line_color, width=2),
            name="Close"
        ))

        # Colorbar (Volume)
        if self.show_colorbar:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    colorscale=self.colorscale,
                    showscale=True,
                    cmin=vol_scaled.min(),
                    cmax=vol_scaled.max(),
                    colorbar=dict(
                        title="log(1 + Volume)" if self.log_scale else "Volume",
                        thickness=15,
                        len=0.8,
                        outlinewidth=0,
                        ticks="outside",
                        x=1.08,          # move colorbar to the right
                        y=0.5,           # center vertically
                        xanchor="left",
                        yanchor="middle"
                    )
                ),
                hoverinfo="none",
                showlegend=False
            ))

        # Layout adjustments
        fig.update_layout(
            title=self.title,
            xaxis_title="Date",
            yaxis_title="Price",
            template=self.template,
            font=dict(size=13),
            legend=dict(
                orientation="h",     # horizontal legend
                yanchor="bottom",
                y=1.02,              # move slightly above plot
                xanchor="left",
                x=0
            ),
            margin=dict(r=120)       # add space on the right for colorbar
        )

        fig.show()
