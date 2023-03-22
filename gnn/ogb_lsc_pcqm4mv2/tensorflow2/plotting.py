# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import wandb
import logging
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd


class Plotting:
    """Class to manage custom plotting results and metrics for wandb + locally."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def set_up_dataframe(preds, truth):
        df = pd.DataFrame(list(zip(preds, truth)), columns=["preds", "truth"])
        df["mae"] = np.abs(preds - truth)
        df = df.sort_values("mae", ignore_index=True)
        df["delta"] = df["mae"] - df["mae"].mean()
        df["rolling_mean"] = df["mae"].expanding().mean()
        return df

    @staticmethod
    def plot_predictions_to_wandb(df):
        """Plot the mae in ascending order to wandb for analysis."""
        fig = px.line(df, y="mae", title="MAE by molecule", log_y=True)
        fig.update_layout(xaxis_title=r"molecule #", yaxis_title=r"MAE")
        fig.add_hline(
            df["mae"].mean(),
            line_width=1,
            line_color="red",
            opacity=0.6,
            line_dash="dot",
            annotation={"text": f"Average: {df['mae'].mean():.3f}"},
        )

        fig.add_hline(
            0.079, line_width=1, line_color="red", opacity=0.6, line_dash="dash", annotation={"text": "Target: 0.079"}
        )
        wandb.log({f"Validation MAE Sorted": fig})

    @staticmethod
    def plot_predictions_histogram(df):
        """Plot the histogram of predictions + truth values"""

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["preds"], name="preds"))
        fig.add_trace(go.Histogram(x=df["truth"], name="truth"))

        # Overlay both histograms
        fig.update_layout(barmode="overlay")
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.5)
        fig.update_layout(xaxis_title=r"HOMO-LUMO", yaxis_title=r"count.")
        wandb.log({f"Predictions Hist": fig})

    @staticmethod
    def plot_mae_histogram(df):
        fig = px.histogram(df, x="mae", nbins=100, title="Mean Absolute Error Distribution")
        fig.update_layout(xaxis_title=r"MAE", yaxis_title=r"count.")
        wandb.log({f"MAE Hist": fig})

    @staticmethod
    def plot_delta_histogram(df):
        fig = px.histogram(df, x="delta", nbins=100, title=r"Prediction Error - Average of Distribution")
        fig.update_layout(xaxis_title=r"$\Delta - \mu$", yaxis_title=r"count.")
        wandb.log({f"Delta Hist": fig})

    @staticmethod
    def plot_rolling_mae(df):
        """Plot the rolling MAE average - number of molecules removed to reach X score."""
        # Sort them in reverse order, makes reading the axis easier
        # Take the worst 3000 results to see the impact clearly
        df = df.sort_values("rolling_mean", ignore_index=True, ascending=False)
        fig = px.line(df, y="rolling_mean", title="MAE with worst molecules removed", log_y=False)
        fig.update_layout(xaxis_title=r"# molecules removed", yaxis_title=r"MAE")
        fig.update_layout(xaxis_range=[0, 4000])
        fig.add_hline(
            df["mae"].mean(),
            line_width=1,
            line_color="red",
            opacity=0.6,
            line_dash="dot",
            annotation={"text": f"Average: {df['mae'].mean():.3f}"},
        )

        fig.add_hline(
            0.079, line_width=1, line_color="red", opacity=0.6, line_dash="dash", annotation={"text": "Target: 0.079"}
        )
        fig.add_vline(500, line_width=1, line_color="gray", line_dash="dash")
        fig.add_vline(1000, line_width=1, line_color="gray", line_dash="dash")
        fig.add_vline(1500, line_width=1, line_color="gray", line_dash="dash")
        fig.add_vline(2000, line_width=1, line_color="gray", line_dash="dash")
        fig.update_xaxes(range=[0, 4000])
        wandb.log({f"Removing mols Impact": fig})

    @staticmethod
    def plot_scatter(df):
        """Plot the preds vs the truth on a 2D scatter"""
        fig = px.scatter(df, x="preds", y="truth", title="Predictions vs True Values", color="mae")
        wandb.log({f"Predictions vs True Values": fig})

        # Poor Results
        df = df.query("mae > 2")
        fig = px.scatter(df, x="preds", y="truth", title="Predictions vs True Values", color="mae")
        wandb.log({f"Predictions vs True Values [mae > 2]": fig})

    def plot_manager(self, preds, truth):
        """Plots to log to wandb:
        1. Sorted Predictions distribution
        2. Impact of removing the worst samples
        3.
        """
        df = self.set_up_dataframe(preds, truth)
        self.plot_predictions_to_wandb(df)
        self.plot_predictions_histogram(df)
        self.plot_mae_histogram(df)
        self.plot_delta_histogram(df)
        self.plot_rolling_mae(df)
        self.plot_scatter(df)
