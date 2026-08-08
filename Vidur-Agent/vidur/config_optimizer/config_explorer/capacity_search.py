import argparse
import glob
import os
import platform
import shlex
from subprocess import Popen

import pandas as pd
import ray

from vidur.config_optimizer.config_explorer.config import JobConfig, SimulationConfig
from vidur.config_optimizer.config_explorer.ray_utils import (
    CpuAssignmentManager,
    get_ip,
)
from vidur.logger import init_logger

logger = init_logger(__name__)


class CapacitySearch:
    def __init__(
        self,
        job_config: JobConfig,
        args: argparse.Namespace,
        cpu_core_assignment_manager: CpuAssignmentManager = None,
        cpu_core_id: int = None,
    ):
        self.node_ip = get_ip()
        self.cpu_core_id = None
        self.job_config = job_config
        self.args = args
        self.cpu_core_assignment_manager = cpu_core_assignment_manager
        self.cpu_core_id = cpu_core_id

    def release_cpu_core_id(self):
        if self.cpu_core_id is None:
            return

        ray.get(
            self.cpu_core_assignment_manager.release_cpu_core_id.remote(
                self.node_ip,
                self.cpu_core_id,
            )
        )

    def _generate_run_command(
        self,
        simulation_config: SimulationConfig,
    ):
        cpu_affinity_command = ""
        if self.cpu_core_id is not None and platform.system() != "Darwin":
            cpu_affinity_command = f"taskset --cpu-list {self.cpu_core_id}"

        command = f"nice -n 1 {cpu_affinity_command} python -m vidur.main {simulation_config.to_args()}"
        logger.debug(f"Running command: {command}")

        return command

    def _is_under_sla(self, run_dir) -> bool:
        boolean_condition = self.job_config.slo_configs.type
        quantile = self.job_config.slo_configs.quantile

        if len(self.job_config.slo_configs.slos) == 1:
            slo_config = self.job_config.slo_configs.slos[0]
            metric_files = glob.glob(f"{run_dir}/*/plots/{slo_config.metric}.csv")
            if len(metric_files) != 1:
                raise ValueError(
                    f"{len(metric_files)} metric files found for {slo_config.metric} in {run_dir}"
                )

            metric_df = pd.read_csv(metric_files[0])
            metric_value = metric_df[slo_config.metric].quantile(quantile)
            return metric_value <= slo_config.value
        else:
            request_metrics_files = glob.glob(f"{run_dir}/*/request_metrics.csv")
            if len(request_metrics_files) != 1:
                raise ValueError(
                    f"{len(request_metrics_files)} request_metrics files found in {run_dir}"
                )

            request_metrics = pd.read_csv(request_metrics_files[0])
            request_metrics["is_under_slo"] = boolean_condition == "and"
            for slo_config in self.job_config.slo_configs.slos:
                if boolean_condition == "and":
                    request_metrics["is_under_slo"] = request_metrics[
                        "is_under_slo"
                    ] & (request_metrics[slo_config.metric] <= slo_config.value)
                else:
                    request_metrics["is_under_slo"] = request_metrics[
                        "is_under_slo"
                    ] | (request_metrics[slo_config.metric] <= slo_config.value)

            return (
                request_metrics["is_under_slo"].sum() / len(request_metrics) >= quantile
            )

    def is_under_sla(self, qps: float, num_replicas: int) -> bool:
        simulator_config = SimulationConfig(
            output_dir=self.args.output_dir,
            cache_dir=self.args.cache_dir,
            time_limit=self.args.time_limit,
            qps=qps,
            num_replicas=num_replicas,
            job_config=self.job_config,
        )
        run_dir = simulator_config.get_run_dir()
        os.makedirs(run_dir, exist_ok=True)

        try:
            is_under_sla = self._is_under_sla(run_dir)
            return is_under_sla
        except Exception as e:
            pass

        with open(f"{run_dir}/output.log", "a") as output_file:
            command = self._generate_run_command(simulator_config)
            # write command to a file
            output_file.write(f"Running command: {command}\n")
            # run and wait on the command
            args = shlex.split(command)
            p = Popen(args, stdout=output_file, stderr=output_file)
            p.wait()

        is_under_sla = self._is_under_sla(run_dir)
        return is_under_sla

    def search_isoqps(self):
        """
        Iterate over the different qps values suggested
        """
        for multiplier in self.job_config.qps_multipliers:
            qps = self.job_config.start_qps * multiplier
            self.is_under_sla(qps, self.job_config.start_num_replicas)

    def search_qps(self):
        """
        Perform binary search to find the maximum QPS under the SLO
        """
        left = 0
        right = self.job_config.start_qps * 2
        is_right_limit_found = False

        for _ in range(self.args.max_iterations):
            if is_right_limit_found:
                # stopping condition - we have reached the minimum granularity
                if abs(left - right) < self.args.min_search_granularity * right / 100:
                    break

                if right < self.args.min_qps:
                    logger.warning(
                        f"Right limit {right} is less than minimum search QPS {self.args.min_qps}. Stopping search."
                    )
                    break

                qps = (left + right) / 2

                is_under_sla = self.is_under_sla(
                    qps, self.job_config.start_num_replicas
                )

                if is_under_sla:
                    left = qps
                else:
                    right = qps
            else:
                is_under_sla = self.is_under_sla(
                    right, self.job_config.start_num_replicas
                )
                if is_under_sla:
                    left = right
                    right *= 2
                else:
                    is_right_limit_found = True
        return left

    def search_num_replicas(self):
        """
        Perform binary search (lower bound) to find the minimum number of replicas required to meet the SLO
        """
        left = 1
        right = 2 * self.job_config.start_num_replicas
        is_right_limit_found = False

        for _ in range(self.args.max_iterations):
            if is_right_limit_found:
                # stopping condition - we can only have whole number of replicas
                if left > right:
                    break

                num_replicas = (left + right) // 2

                is_under_sla = self.is_under_sla(
                    self.job_config.start_qps, num_replicas
                )

                if not is_under_sla:
                    left = num_replicas + 1
                else:
                    right = num_replicas - 1
            else:
                is_under_sla = self.is_under_sla(self.job_config.start_qps, right)
                if not is_under_sla:
                    left = right + 1
                    right *= 2
                else:
                    is_right_limit_found = True
        return left

    def search(self):
        try:
            logger.info(
                f"Starting search for {self.job_config.get_human_readable_name()}",
            )
            if self.job_config.search_for == "qps":
                max_qps_under_sla = self.search_qps()
                logger.info(
                    f"Max QPS under SLO for {self.job_config.get_human_readable_name()}: {max_qps_under_sla}",
                )
            elif self.job_config.search_for == "num_replicas":
                min_num_replicas_under_slo = self.search_num_replicas()
                logger.info(
                    f"Min number of replicas under SLO for {self.job_config.get_human_readable_name()}: {min_num_replicas_under_slo}",
                )
            elif self.job_config.search_for == "isoqps":
                self.search_isoqps()
                logger.info(
                    f"Finished isoqps runs for {self.job_config.get_human_readable_name()}",
                )
        except Exception as e:
            logger.error(
                f"Error running: {self.job_config.get_human_readable_name()}, failed with error: {e}",
            )
        finally:
            self.release_cpu_core_id()
