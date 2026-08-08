from enum import IntEnum


class BaseIntEnum(IntEnum):
    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, string):
        if not isinstance(string, str):
            string = str(string)
        return cls[string.upper()]
