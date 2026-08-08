from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from ddsketch.ddsketch import DDSketch
from matplotlib import pyplot as plt

from vidur.logger import init_logger

logger = init_logger(__name__)


class CDFSketch:
    def __init__(
        self,
        metric_name: str,
        save_table_to_wandb: bool = True,
        save_plots: bool = True,
    ) -> None:
        # metrics are a data series of two-dimensional (x, y) datapoints
        self._sketch = DDSketch(relative_accuracy=0.01)
        # column name
        self._metric_name = metric_name

        # most recently collected y datapoint for incremental updates
        # to aid incremental updates to y datapoints
        self._last_data = 0

        self._save_table_to_wandb = save_table_to_wandb
        self._save_plots = save_plots

    def __len__(self):
        return int(self._sketch.count)

    @property
    def sum(self) -> float:
        return self._sketch.sum

    # add a new x, y datapoint
    def put(self, data: float) -> None:
        self._last_data = data
        self._sketch.add(data)

    # add a new datapoint as an incremental (delta) update to
    # recently collected datapoint
    def put_delta(self, delta: float) -> None:
        data = self._last_data + delta
        self.put(data)

    def to_df(self) -> pd.DataFrame:
        # get quantiles at 1% intervals
        quantiles = np.linspace(0, 1, 101)
        # get quantile values
        quantile_values = [self._sketch.get_quantile_value(q) for q in quantiles]
        # create dataframe
        df = pd.DataFrame({"cdf": quantiles, self._metric_name: quantile_values})

        return df

    def _save_df(self, df: pd.DataFrame, path: str, plot_name: str) -> None:
        df.to_csv(f"{path}/{plot_name}.csv", index=False)

        if wandb.run and self._save_table_to_wandb:
            wand_table = wandb.Table(dataframe=df)
            wandb.log({f"{plot_name}_table": wand_table}, step=0)

    def print_distribution_stats(self, plot_name: str) -> None:
        if self._sketch._count == 0:
            return

        logger.debug(
            f"{plot_name}: {self._metric_name} stats:"
            f" min: {self._sketch._min},"
            f" max: {self._sketch._max},"
            f" mean: {self._sketch.avg},"
            f" 25th percentile: {self._sketch.get_quantile_value(0.25)},"
            f" median: {self._sketch.get_quantile_value(0.5)},"
            f" 75th percentile: {self._sketch.get_quantile_value(0.75)},"
            f" 95th percentile: {self._sketch.get_quantile_value(0.95)},"
            f" 99th percentile: {self._sketch.get_quantile_value(0.99)}"
            f" 99.9th percentile: {self._sketch.get_quantile_value(0.999)}"
            f" count: {self._sketch._count}"
            f" sum: {self._sketch.sum}"
        )
        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_min": self._sketch._min,
                    f"{plot_name}_max": self._sketch._max,
                    f"{plot_name}_mean": self._sketch.avg,
                    f"{plot_name}_25th_percentile": self._sketch.get_quantile_value(
                        0.25
                    ),
                    f"{plot_name}_median": self._sketch.get_quantile_value(0.5),
                    f"{plot_name}_75th_percentile": self._sketch.get_quantile_value(
                        0.75
                    ),
                    f"{plot_name}_95th_percentile": self._sketch.get_quantile_value(
                        0.95
                    ),
                    f"{plot_name}_99th_percentile": self._sketch.get_quantile_value(
                        0.99
                    ),
                    f"{plot_name}_99.9th_percentile": self._sketch.get_quantile_value(
                        0.999
                    ),
                    f"{plot_name}_count": self._sketch.count,
                    f"{plot_name}_sum": self._sketch.sum,
                },
                step=0,
            )

    def plot_cdf(self, path: str, plot_name: str, x_axis_label: str = None) -> None:
        if self._sketch._count == 0:
            return

        if x_axis_label is None:
            x_axis_label = self._metric_name

        df = self.to_df()

        self.print_distribution_stats(plot_name)

        if wandb.run:
            wandb_df = df.copy()
            # rename the self._metric_name column to x_axis_label
            wandb_df = wandb_df.rename(columns={self._metric_name: x_axis_label})

            wandb.log(
                {
                    f"{plot_name}_cdf": wandb.plot.line(
                        wandb.Table(dataframe=wandb_df),
                        "cdf",
                        x_axis_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        if self._save_plots:
            plt.clf()
            sns.lineplot(df, x="cdf", y=self._metric_name)
            plt.savefig(f"{path}/{plot_name}.png")

        self._save_df(df, path, plot_name)

    @staticmethod
    def plot_cdfs(
        cdf_sketches: Dict[int, "CDFSketch"],
        path: str,
        plot_name: str,
        name: str,
        y_axis_label: str = None,
        save_plot: bool = True,
    ) -> None:
        metric_name = next(iter(cdf_sketches.values()), None)._metric_name
        dfs = [cdf_sketch.to_df() for cdf_sketch in cdf_sketches.values()]
        for key, df in zip(cdf_sketches.keys(), dfs):
            df[name] = key
        df = pd.concat(dfs)

        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_cdf": wandb.plot.line_series(
                        xs=[df["cdf"].values for df in dfs],
                        ys=[df[metric_name].values for df in dfs],
                        keys=list(cdf_sketches.keys()),
                        title=plot_name,
                    )
                },
                step=0,
            )
        if save_plot:
            plt.figure(figsize=(10, 7)) # Increase size for better visibility
            sns.set_style("whitegrid") # Add grid for better readability
            ax =sns.lineplot(df, x="cdf", y=metric_name, hue=name, palette="turbo")
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
