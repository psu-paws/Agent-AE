from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType, NodeSKUType

logger = init_logger(__name__)


@dataclass
class BaseNodeSKUConfig(BaseFixedConfig):
    num_devices_per_node: int
    inter_node_bandwidth_gbps: float = 0.0


@dataclass
class A40PairwiseNvlinkNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.A40
    num_devices_per_node: int = 8

    @staticmethod
    def get_type():
        return NodeSKUType.A40_PAIRWISE_NVLINK


@dataclass
class A100PairwiseNvlinkNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.A100
    num_devices_per_node: int = 4
    inter_node_bandwidth_gbps: float = 100.0  # 4 x 200Gbps = 4 x 25 GB/s

    @staticmethod
    def get_type():
        return NodeSKUType.A100_PAIRWISE_NVLINK


@dataclass
class H100PairwiseNvlinkNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.H100
    num_devices_per_node: int = 4
    inter_node_bandwidth_gbps: float = 200.0  # 4 x 400Gbps = 4 x 50 GB/s

    @staticmethod
    def get_type():
        return NodeSKUType.H100_PAIRWISE_NVLINK


@dataclass
class A100DgxNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.A100
    num_devices_per_node: int = 8
    inter_node_bandwidth_gbps: float = 200.0  # 8 x 200Gbps = 8 x 25 GB/s

    @staticmethod
    def get_type():
        return NodeSKUType.A100_DGX


@dataclass
class A100PcieNodeSKUConfig(BaseNodeSKUConfig):
    """8x A100 80GB PCIe, For validation"""

    device_sku_type: DeviceSKUType = DeviceSKUType.A100
    num_devices_per_node: int = 8
    # Single-node SKU: 0 disables the analytical cross-node KV transfer estimate.
    inter_node_bandwidth_gbps: float = 0.0

    @staticmethod
    def get_type():
        return NodeSKUType.A100_PCIE


@dataclass
class A100NvlinkPcieNodeSKUConfig(BaseNodeSKUConfig):
    """8x A100 80GB with an NVLink tensor-parallel group and a PCIe KV handoff."""

    device_sku_type: DeviceSKUType = DeviceSKUType.A100
    num_devices_per_node: int = 8
    # Used by the analytical cross-node KV estimate, which divides this by
    # num_devices_per_node to get per-GPU bandwidth. 170 / 8 = 21.25 GB/s, the
    # rate measured for send_recv on this fabric, so a handoff between replica
    # groups is charged the same bandwidth as one inside a group.
    inter_node_bandwidth_gbps: float = 170.0

    @staticmethod
    def get_type():
        return NodeSKUType.A100_NVLINK_PCIE


@dataclass
class H100DgxNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.H100
    num_devices_per_node: int = 8
    inter_node_bandwidth_gbps: float = 400.0  # 8 x 400Gbps = 8 x 50 GB/s

    @staticmethod
    def get_type():
        return NodeSKUType.H100_DGX
