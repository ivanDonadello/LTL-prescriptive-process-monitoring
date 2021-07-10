from enum import Enum


class ConstraintChecker(Enum):
    RESPONDED_EXISTENCE = "Responded Existence"
    RESPONSE = "Response"
    ALTERNATE_RESPONSE = "Alternate Response"
    CHAIN_RESPONSE = "Chain Response"
    PRECEDENCE = "Precedence"
    ALTERNATE_PRECEDENCE = "Alternate Precedence"
    CHAIN_PRECEDENCE = "Chain Precedence"
    NOT_RESPONDED_EXISTENCE = "Not Responded Existence"
    NOT_RESPONSE = "Not Response"
    NOT_CHAIN_RESPONSE = "Not Chain Response"
    NOT_PRECEDENCE = "Not Precedence"
    NOT_CHAIN_PRECEDENCE = "Not Chain Precedence"