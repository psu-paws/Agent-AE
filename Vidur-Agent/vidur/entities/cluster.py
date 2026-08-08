import json
from typing import Dict

from vidur.config import BaseRequestGeneratorConfig, ClusterConfig, MetricsConfig
from vidur.entities.base_entity import BaseEntity
from vidur.entities.replica import Replica
from vidur.logger import init_logger
from vidur.types.replica_id import ReplicaId
from vidur.utils.json_encoder import JsonEncoder

logger = init_logger(__name__)


class Cluster(BaseEntity):
    def __init__(
        self,
        cluster_config: ClusterConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        self._id = Cluster.generate_id()
        self._config = cluster_config

        # get metrics config
        self._output_dir = metrics_config.output_dir

        # Init replica object handles
        self._replicas: Dict[ReplicaId, Replica] = {}

        for i in range(self._config.num_replicas):
            replica = Replica(
                replica_config=self._config.get_replica_config(i),
                role=self._config.get_replica_role(i),
            )
            self._replicas[replica.id] = replica

        if metrics_config.write_json_trace:
            self._write_cluster_info_to_file()

    @property
    def replicas(self):
        return self._replicas

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "num_replicas": len(self._replicas),
            "replicas": [replica.to_dict() for replica in self._replicas.values()],
        }

    def _write_cluster_info_to_file(self) -> None:
        cluster_info = {
            "id": self._id,
            "num_replicas": len(self._replicas),
            "replicas": [replica.to_dict() for replica in self._replicas.values()],
        }

        cluster_file = f"{self._output_dir}/cluster.json"
        with open(cluster_file, "w") as f:
            json.dump(cluster_info, f, cls=JsonEncoder)
