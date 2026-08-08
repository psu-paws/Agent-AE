import json

from vidur.types.event_type import EventType
from vidur.types.replica_id import ReplicaId


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ReplicaId):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, EventType):
            return str(obj)
        return json.JSONEncoder.default(self, obj)
