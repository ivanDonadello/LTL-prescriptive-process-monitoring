from enum import Enum, auto


class LabelType(Enum):
    DEFAULT = auto()
    TRACE_DURATION = auto()
    TRACE_NUMERICAL_ATTRIBUTES = auto()
