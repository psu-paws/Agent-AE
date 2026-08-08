from dataclasses import dataclass


@dataclass(frozen=True)
class ReplicaId:
    id: int

    def __str__(self):
        return str(self.id)

    def __lt__(self, other: "ReplicaId"):
        return self.id < other.id
