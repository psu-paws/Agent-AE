import atexit
import heapq
import json
import zipfile
from typing import List

import wandb

from vidur.config import SimulationConfig
from vidur.entities import Cluster
from vidur.events import BaseEvent, RequestArrivalEvent
from vidur.logger import init_logger
from vidur.metrics.cluster_metrics_store import ClusterMetricsStore
from vidur.request_generator import RequestGeneratorRegistry
from vidur.scheduler import GlobalSchedulerRegistry
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.utils.json_encoder import JsonEncoder
from vidur.types import EventType

logger = init_logger(__name__)


class Simulator:
    def __init__(self, config: SimulationConfig) -> None:
        self._config: SimulationConfig = config

        self._time = 0
        self._time_limit_reached = False
        self._time_limit = self._config.time_limit
        if not self._time_limit:
            self._time_limit = float("inf")

        self._event_queue: List[BaseEvent] = []

        self._event_trace = []
        self._event_chrome_trace = []

        self._cluster = Cluster(
            cluster_config=self._config.cluster_config,
            metrics_config=self._config.metrics_config,
        )
        self._request_generator = RequestGeneratorRegistry.get(
            self._config.request_generator_config.get_type(),
            self._config.request_generator_config,
        )
        # Per-session metric stores are only meaningful for multi-turn traces.
        session_ids = (
            list(range(self._request_generator.num_sessions))
            if self._request_generator.is_multi_turn
            else []
        )
        self._cluster_metric_store = ClusterMetricsStore(
            simulation_config=self._config,
            replicas=self._cluster.replicas,
            requests=session_ids,
        )
        self._scheduler = GlobalSchedulerRegistry.get(
            self._config.cluster_config.global_scheduler_config.get_type(),
            self._config,
            self._cluster.replicas,
        )

        self._init_event_queue()
        atexit.register(self._write_output)

    def run(self) -> None:
        logger.info(f"Starting simulation with cluster: {self._cluster}")

        while not self._time_limit_reached and self._event_queue:
            event = self._event_queue[0]
            next_event_time = event._time
            next_event_type = event._event_type
            heapq.heappop(self._event_queue)
            self._set_time(event._time)
            new_events = event.handle_event(self._scheduler, self._cluster_metric_store)
            self._add_events(new_events)

            # A turn just finished, so release whatever the trace says follows it
            # within that same session. Traces come in two routes.
            if next_event_type == EventType.REQUEST_END:
                idx = event._request._session_id
                if self._request_generator.has_deps:
                    # Route 1 - dependency graph. A `dep` column names the turns
                    # that must all complete before a given turn may start, so a
                    # completion can unblock several children at once. Each child
                    # arrives once its "last" dependency finishes.
                    completed_turn = event._request.turn_id
                    for request, arrival_time in self._request_generator.get_dep_ready_requests(
                        idx, completed_turn, next_event_time
                    ):
                        self._add_event(RequestArrivalEvent(arrival_time, request))
                else:
                    # Route 2 - simple chain. Turn k+1 follows turn k after the
                    # trace's inter_request_latency (tool call).
                    next_arrival_time = (
                        next_event_time + event._request.inter_request_latency
                    )
                    if (
                        self._request_generator.get_next_request_arrival_time(
                            idx, next_arrival_time
                        )
                        is not None
                    ):
                        self._add_event(
                            RequestArrivalEvent(
                                next_arrival_time,
                                self._request_generator.get_next_request(idx),
                            )
                        )

            if self._config.metrics_config.write_json_trace:
                self._event_trace.append(event.to_dict())

            if self._config.metrics_config.enable_chrome_trace:
                chrome_trace = event.to_chrome_trace()
                if chrome_trace:
                    self._event_chrome_trace.append(chrome_trace)

        assert self._scheduler.is_empty() or self._time_limit_reached

        logger.info(f"Simulation ended at: {self._time}s")

    def _write_output(self) -> None:
        logger.info("Writing output")

        self._cluster_metric_store.plot(self._time)
        logger.info("Metrics written")
        self._write_eviction_metrics()

        if self._config.metrics_config.write_json_trace:
            self._write_event_trace()
            logger.info("Json event trace written")

        if self._config.metrics_config.enable_chrome_trace:
            self._write_chrome_trace()
            logger.info("Chrome event trace written")

    def _add_event(self, event: BaseEvent) -> None:
        heapq.heappush(self._event_queue, event)

    def _add_events(self, events: List[BaseEvent]) -> None:
        for event in events:
            self._add_event(event)

    def _init_event_queue(self) -> None:
        for idx in range(self._request_generator.num_sessions):
            request = self._request_generator.get_next_request(idx)
            if request is not None:
                self._add_event(RequestArrivalEvent(request.arrived_at, request))

    def _set_time(self, time: float) -> None:
        self._time = time
        if self._time > self._time_limit:
            logger.info(
                f"Time limit reached: {self._time_limit}s terminating the simulation."
            )
            self._time_limit_reached = True

    def _write_eviction_metrics(self) -> None:
        output_dir = self._config.metrics_config.output_dir
        for replica_id, replica_sched in self._scheduler._replica_schedulers.items():
            if not hasattr(replica_sched, '_kv_cache_manager'):
                continue
            block_pool = replica_sched._kv_cache_manager.block_pool
            metrics = block_pool.get_eviction_metrics()

            sum_pf = getattr(replica_sched, 'sum_prefill_tokens', 0)
            sum_hit = getattr(replica_sched, 'sum_kvhit_tokens', 0)
            token_hit_rate = sum_hit / sum_pf if sum_pf > 0 else 0.0
            metrics['sum_prefill_tokens'] = sum_pf
            metrics['sum_kvhit_tokens'] = sum_hit
            metrics['token_cache_hit_rate'] = token_hit_rate
            # Decode admission gate: how often it actually held a request back.
            metrics['decode_admissions'] = getattr(
                replica_sched, 'num_decode_admissions', 0)
            metrics['decode_gate_holds'] = getattr(
                replica_sched, 'num_decode_gate_holds', 0)

            logger.info(
                f"Replica {replica_id} eviction metrics: "
                f"evictions={metrics['num_evictions']}, "
                f"token_hit_rate={token_hit_rate:.4f} "
                f"({sum_hit}/{sum_pf} tokens), "
                f"unique_evicted_hashes={metrics['num_unique_evicted_hashes']}, "
                f"decode_admissions={metrics['decode_admissions']}, "
                f"decode_gate_holds={metrics['decode_gate_holds']}"
            )
            # Write summary JSON per replica
            metrics_path = f"{output_dir}/eviction_metrics_replica_{replica_id}.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
        logger.info("Eviction metrics written")

    def _write_event_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/event_trace.json"
        with open(trace_file, "w") as f:
            json.dump(self._event_trace, f, cls=JsonEncoder)

    def _write_chrome_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/chrome_trace.json"

        chrome_trace = {"traceEvents": self._event_chrome_trace}

        with open(trace_file, "w") as f:
            json.dump(chrome_trace, f, cls=JsonEncoder)

        if wandb.run:
            zip_file_path = f"{self._config.output_dir}/chrome_trace.zip"
            with zipfile.ZipFile(
                zip_file_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                zf.writestr(
                    "chrome_trace.json",
                    json.dumps(chrome_trace, cls=JsonEncoder),
                )
            wandb.save(zip_file_path, policy="now")
