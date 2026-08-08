from vidur.scheduler.request_queue.edf_request_queue import EDFRequestQueue
from vidur.scheduler.request_queue.fcfs_request_queue import FCFSRequestQueue
from vidur.types import RequestQueueType
from vidur.utils.base_registry import BaseRegistry


class RequestQueueRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> RequestQueueType:
        return RequestQueueType.from_str(key_str)


RequestQueueRegistry.register(RequestQueueType.FCFS, FCFSRequestQueue)
RequestQueueRegistry.register(RequestQueueType.EDF, EDFRequestQueue)
