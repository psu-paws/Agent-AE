from vidur.entities.request import Request


class PrioritizedRequest:
    def __init__(self, request: Request, priority: float):
        self._request = request
        self._priority = priority

    @property
    def request(self):
        return self._request

    @property
    def priority(self):
        return self._priority

    def __lt__(self, other):
        assert isinstance(other, PrioritizedRequest)
        if self.priority == other.priority:
            return self.request.id < other.request.id
        return self.priority < other.priority
