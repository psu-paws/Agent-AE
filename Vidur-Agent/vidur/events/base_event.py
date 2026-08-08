from abc import ABC, abstractmethod
from typing import List

from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType


class BaseEvent(ABC):
    _id = 0

    def __init__(self, time: float, event_type: EventType):
        self._time = time
        self._id = BaseEvent.generate_id()
        self._event_type = event_type
        self._priority_number = self._get_priority_number()

    @classmethod
    def generate_id(cls):
        cls._id += 1
        return cls._id

    @property
    def id(self) -> int:
        return self._id

    @property
    def time(self):
        return self._time

    @property
    def event_type(self):
        return self._event_type

    @abstractmethod
    def handle_event(
        self,
        current_time: float,
        scheduler: BaseGlobalScheduler,
        metrics_store: ClusterMetricsStore,
    ) -> List["BaseEvent"]:
        pass

    """
        We give the highest priority to the event with the lowest time.
        Then request arrival events are given priority over other events.
        Finally, we give priority to the event with the lowest id for causality.
    """

    def _get_priority_number(self):
        return (self._time, self.event_type, self._id)

    def __lt__(self, other):
        assert isinstance(other, BaseEvent)
        return self._priority_number < other._priority_number

    def __gt__(self, other):
        assert isinstance(other, BaseEvent)
        return self._priority_number > other._priority_number

    def __eq__(self, other):
        assert isinstance(other, BaseEvent)
        if self._id == other._id:
            assert self._time == other._time and self._event_type == other._event_type
            return True
        return False

    def __str__(self) -> str:
        # use to_dict to get a dict representation of the object
        # and convert it to a string
        class_name = self.__class__.__name__
        return f"{class_name}({str(self.to_dict())})"

    def to_dict(self):
        return {"time": self.time, "event_type": self.event_type}

    def to_chrome_trace(self) -> dict:
        return None
