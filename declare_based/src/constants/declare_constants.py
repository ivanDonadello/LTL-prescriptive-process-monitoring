from declare_based.src.declare_templates.log import existence as dt_existence_log
from declare_based.src.declare_templates.log import choice as dt_choice_log
from declare_based.src.declare_templates.log import relation as dt_relation_log
from declare_based.src.declare_templates.log import negative_relation as dt_negative_relation_log
from declare_based.src.declare_templates.trace import relation as dt_relation_trace
from declare_based.src.declare_templates.trace import negative_relation as dt_negative_relation_trace

EXISTENCE = "Existence"
ABSENCE = "Absence"
INIT = "Init"
EXACTLY = "Exactly"

CHOICE = "Choice"
EXCLUSIVE_CHOICE = "Exclusive Choice"

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


DT_LOG_METHODS = {
    INIT: dt_existence_log.mp_init,
    EXISTENCE: dt_existence_log.mp_existence,
    ABSENCE: dt_existence_log.mp_absence,
    EXACTLY: dt_existence_log.mp_exactly,
    CHOICE: dt_choice_log.mp_choice,
    EXCLUSIVE_CHOICE: dt_choice_log.mp_exclusive_choice,
    RESPONDED_EXISTENCE: dt_relation_log.mp_responded_existence,
    RESPONSE: dt_relation_log.mp_response,
    ALTERNATE_RESPONSE: dt_relation_log.mp_alternate_response,
    CHAIN_RESPONSE: dt_relation_log.mp_chain_response,
    PRECEDENCE: dt_relation_log.mp_precedence,
    ALTERNATE_PRECEDENCE: dt_relation_log.mp_alternate_precedence,
    CHAIN_PRECEDENCE: dt_relation_log.mp_chain_precedence,
    NOT_RESPONDED_EXISTENCE: dt_negative_relation_log.mp_not_responded_existence,
    NOT_RESPONSE: dt_negative_relation_log.mp_not_response,
    NOT_CHAIN_RESPONSE: dt_negative_relation_log.mp_not_chain_response,
    NOT_PRECEDENCE: dt_negative_relation_log.mp_not_precedence,
    NOT_CHAIN_PRECEDENCE: dt_negative_relation_log.mp_not_chain_precedence
}

DT_TRACE_METHODS = {
    RESPONDED_EXISTENCE: dt_relation_trace.mp_responded_existence,
    RESPONSE: dt_relation_trace.mp_response,
    ALTERNATE_RESPONSE: dt_relation_trace.mp_alternate_response,
    CHAIN_RESPONSE: dt_relation_trace.mp_chain_response,
    PRECEDENCE: dt_relation_trace.mp_precedence,
    ALTERNATE_PRECEDENCE: dt_relation_trace.mp_alternate_precedence,
    CHAIN_PRECEDENCE: dt_relation_trace.mp_chain_precedence,
    NOT_RESPONDED_EXISTENCE: dt_negative_relation_trace.mp_not_responded_existence,
    NOT_RESPONSE: dt_negative_relation_trace.mp_not_response,
    NOT_CHAIN_RESPONSE: dt_negative_relation_trace.mp_not_chain_response,
    NOT_PRECEDENCE: dt_negative_relation_trace.mp_not_precedence,
    NOT_CHAIN_PRECEDENCE: dt_negative_relation_trace.mp_not_chain_precedence
}
