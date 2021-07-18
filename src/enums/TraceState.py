from enum import Enum


class TraceState(Enum):
    VIOLATED = 0
    SATISFIED = 1
    POSSIBLY_VIOLATED = 2
    POSSIBLY_SATISFIED = 3
