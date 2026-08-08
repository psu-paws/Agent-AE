from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import pyplot as plt

from vidur.logger import init_logger

logger = init_logger(__name__)


class DataSeries:
    def __init__(
        self,
        x_name: str,
        y_name: str,
        subsamples: Optional[int] = None,
        save_table_to_wandb: bool = True,
        save_plots: bool = True,
    ) -> None:
        # metrics are a data series of two-dimensional (x, y) datapoints
        self._data_series = []
        # column names of x, y datatpoints for data collection
        self._x_name = x_name
        self._y_name = y_name

        # most recently collected y datapoint for incremental updates
        # to aid incremental updates to y datapoints
        self._last_data_y = 0

        self._subsamples = subsamples
        self._save_table_to_wandb = save_table_to_wandb
        self._save_plots = save_plots

    def __len__(self):
        return len(self._data_series)

    @property
    def _metric_name(self) -> str:
        return self._y_name

    # add a new x, y datapoint
    def put(self, data_x: float, data_y: float) -> None:
        self._last_data_y = data_y
        self._data_series.append((data_x, data_y))

    # add a new x, y datapoint as an incremental (delta) update to
    # recently collected y datapoint
    def put_delta(self, data_x: float, data_y_delta: float) -> None:
        data_y = self._last_data_y + data_y_delta
        self._last_data_y = data_y
        self.put(data_x, data_y)

    # convert list of x, y datapoints to a pandas dataframe
    def to_df(self):
        return pd.DataFrame(self._data_series, columns=[self._x_name, self._y_name])

    def to_quantile_df(self):
        df = self.to_df()
        quantiles = np.linspace(0, 1, 101)
        quantile_values = df[self._y_name].quantile(quantiles)

        return pd.DataFrame({"cdf": quantiles, self._y_name: quantile_values})

    def get_stats(self, y_name: str = None):
        if len(self._data_series) == 0:
            return

        if y_name is None:
            y_name = self._y_name

        df = self.to_df()
        return {
            "min": df[y_name].min(),
            "max": df[y_name].max(),
            "mean": df[y_name].mean(),
            "median": df[y_name].median(),
            "95th_percentile": df[y_name].quantile(0.95),
            "99th_percentile": df[y_name].quantile(0.99),
            "99.9th_percentile": df[y_name].quantile(0.999),
            "sum": df[y_name].sum(),
        }

    def print_distribution_stats(
        self, df: pd.DataFrame, plot_name: str, y_name: str = None
    ) -> None:
        if len(self._data_series) == 0:
            return

        if y_name is None:
            y_name = self._y_name

        logger.debug(
            f"{plot_name}: {y_name} stats:"
            f" min: {df[y_name].min()},"
            f" max: {df[y_name].max()},"
            f" mean: {df[y_name].mean()},"
            f" median: {df[y_name].median()},"
            f" 95th percentile: {df[y_name].quantile(0.95)},"
            f" 99th percentile: {df[y_name].quantile(0.99)}"
            f" 99.9th percentile: {df[y_name].quantile(0.999)}"
        )
        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_min": df[y_name].min(),
                    f"{plot_name}_max": df[y_name].max(),
                    f"{plot_name}_mean": df[y_name].mean(),
                    f"{plot_name}_median": df[y_name].median(),
                    f"{plot_name}_95th_percentile": df[y_name].quantile(0.95),
                    f"{plot_name}_99th_percentile": df[y_name].quantile(0.99),
                    f"{plot_name}_99.9th_percentile": df[y_name].quantile(0.999),
                },
                step=0,
            )

    def _save_df(self, df: pd.DataFrame, path: str, plot_name: str) -> None:
        df.to_csv(f"{path}/{plot_name}.csv", index=False)
        if wandb.run and self._save_table_to_wandb:
            wand_table = wandb.Table(dataframe=df)
            wandb.log({f"{plot_name}_table": wand_table}, step=0)

    def plot_scatter(
        self,
        path: str,
        plot_name: str,
        y_axis_label: str = None,
    ) -> None:
        if len(self._data_series) == 0:
            return

        if y_axis_label is None:
            y_axis_label = self._y_name

        df = self.to_df()
        self.print_distribution_stats(df, plot_name)

        if wandb.run:
            wandb_df = df.copy()
            # rename the self._y_name column to y_axis_label
            wandb_df = wandb_df.rename(columns={self._y_name: y_axis_label})

            wandb.log(
                {
                    f"{plot_name}_scatter": wandb.plot.scatter(
                        wandb.Table(dataframe=wandb_df),
                        self._x_name,
                        y_axis_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        if self._save_plots:
            plt.clf()
            sns.scatterplot(df, x=self._x_name, y=self._y_name)
            plt.savefig(f"{path}/{plot_name}.png")

        self._save_df(df, path, plot_name)

    def plot_step(
        self,
        path: str,
        plot_name: str,
        y_axis_label: str = None,
        start_time: float = 0,
        y_cumsum: bool = True,
    ) -> None:

        if len(self._data_series) == 0:
            return

        if y_axis_label is None:
            y_axis_label = self._y_name

        df = self.to_df()
        df[self._x_name] -= start_time

        if y_cumsum:
            df[self._y_name] = df[self._y_name].cumsum()

        self.print_distribution_stats(df, plot_name)

        if wandb.run:
            wandb_df = df.copy()
            # rename the self._y_name column to y_axis_label
            wandb_df = wandb_df.rename(columns={self._y_name: y_axis_label})

            wandb.log(
                {
                    f"{plot_name}_step": wandb.plot.line(
                        wandb.Table(dataframe=wandb_df),
                        self._x_name,
                        y_axis_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        if self._save_plots:
            plt.clf()
            sns.lineplot(df, x=self._x_name, y=self._y_name)
            plt.savefig(f"{path}/{plot_name}.png")

        self._save_df(df, path, plot_name)

    def plot_cdf(self, path: str, plot_name: str, y_axis_label: str = None) -> None:
        if len(self._data_series) == 0:
            return

        if y_axis_label is None:
            y_axis_label = self._y_name

        df = self.to_quantile_df()

        if wandb.run:
            wandb_df = df.copy()
            # rename the self._y_name column to y_axis_label
            wandb_df = wandb_df.rename(columns={self._y_name: y_axis_label})

            wandb.log(
                {
                    f"{plot_name}_cdf": wandb.plot.line(
                        wandb.Table(dataframe=wandb_df),
                        "cdf",
                        y_axis_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        if self._save_plots:
            plt.clf()
            sns.lineplot(df, x="cdf", y=self._y_name)
            plt.savefig(f"{path}/{plot_name}.png")

        self._save_df(df, path, plot_name)

    def plot_histogram(self, path: str, plot_name: str, bin_count: int) -> None:
        if len(self._data_series) == 0:
            return

        df = self.to_df()
        self.print_distribution_stats(df, plot_name)

        if wandb.run:
            # wandb histogram is highly inaccurate so we need to generate the histogram
            # ourselves and then use wandb bar chart
            histogram_df = (
                df[self._y_name].value_counts(bins=bin_count, sort=False).sort_index()
            )
            histogram_df = histogram_df.reset_index()
            histogram_df.columns = ["Bins", "count"]
            histogram_df["Bins"] = histogram_df["Bins"].apply(lambda x: x.mid)
            histogram_df = histogram_df.sort_values(by=["Bins"])
            # convert to percentage
            histogram_df["Percentage"] = histogram_df["count"] * 100 / len(df)
            # drop bins with less than 0.1% of the total count
            histogram_df = histogram_df[histogram_df["Percentage"] > 0.1]

            wandb.log(
                {
                    f"{plot_name}_histogram": wandb.plot.bar(
                        wandb.Table(dataframe=histogram_df),
                        "Bins",
                        "Percentage",  # wandb plots are horizontal
                        title=plot_name,
                    )
                },
                step=0,
            )

        if self._save_plots:
            plt.clf()
            sns.displot(
                df,
                x=self._y_name,
            )
            plt.savefig(f"{path}/{plot_name}.png")

    @staticmethod
    def plot_cdfs(
        dataseries_dict: Dict[int, "DataSeries"],
        path: str,
        plot_name: str,
        name: str,
        y_axis_label: str = None,
        save_plot: bool = True,
    ) -> None:
        # Get any of the values from the dict
        y_name = next(iter(dataseries_dict.values()), None)._y_name

        dfs = [dataseries.to_quantile_df() for dataseries in dataseries_dict.values()]
        for key, df in zip(dataseries_dict.keys(), dfs):
            df[name] = key
        df = pd.concat(dfs)

        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_cdf": wandb.plot.line_series(
                        xs=[df["cdf"].values for df in dfs],
                        ys=[df[y_name].values for df in dfs],
                        keys=list(dataseries_dict.keys()),
                        title=plot_name,
                    )
                },
                step=0,
            )
        if save_plot:
            plt.clf()
            sns.lineplot(df, x="cdf", y=y_name, hue=name)
            plt.savefig(f"{path}/{plot_name}.png")
        df.to_csv(f"{path}/{plot_name}.csv", index=False)

    @staticmethod
    def plot_steps(
        dataseries_dict: Dict[int, "DataSeries"],
        path: str,
        plot_name: str,
        name: str,
        y_axis_label: str = None,
        start_val: Optional[float] = None,
        y_cumsum: bool = True,
        save_plot: bool = True,
    ) -> None:
        y_name = next(iter(dataseries_dict.values()), None)._y_name
        x_name = next(iter(dataseries_dict.values()), None)._x_name

        dfs = [dataseries.to_df() for dataseries in dataseries_dict.values()]
        for key, df in zip(dataseries_dict.keys(), dfs):
            if start_val is not None:
                df[x_name] -= start_val
            if y_cumsum:
                df[y_name] = df[y_name].cumsum()
            df[name] = key
        df = pd.concat(dfs)

        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_step": wandb.plot.line_series(
                        xs=[df[x_name].values for df in dfs],
                        ys=[df[y_name].values for df in dfs],
                        keys=list(dataseries_dict.keys()),
                        title=plot_name,
                    )
                },
                step=0,
            )
        if save_plot:
            plt.figure(figsize=(10, 7)) # Increase size for better visibility
            sns.set_style("whitegrid") # Add grid for better readability
            ax =sns.lineplot(df, x=x_name, y=y_name, hue=name, palette="turbo")
            sns.move_legend(
                ax, "lower center",
                bbox_to_anchor=(.5, 1.02), # Coordinates: middle of x, just above y
                ncol=5, 
                title=name, 
                frameon=False
            )
            plt.tight_layout()
            plt.savefig(f"{path}/{plot_name}.png", bbox_inches='tight')
            plt.close()

        df.to_csv(f"{path}/{plot_name}.csv", index=False)
