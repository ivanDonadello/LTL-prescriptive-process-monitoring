from src.enums.ConstraintChecker import ConstraintChecker
from src.constraint_checkers import relation as dt_relation_trace
from src.constraint_checkers import negative_relation as dt_negative_relation_trace

DT_TRACE_METHODS = {
    ConstraintChecker.RESPONDED_EXISTENCE.value: dt_relation_trace.mp_responded_existence,
    ConstraintChecker.RESPONSE.value: dt_relation_trace.mp_response,
    ConstraintChecker.ALTERNATE_RESPONSE.value: dt_relation_trace.mp_alternate_response,
    ConstraintChecker.CHAIN_RESPONSE.value: dt_relation_trace.mp_chain_response,
    ConstraintChecker.PRECEDENCE.value: dt_relation_trace.mp_precedence,
    ConstraintChecker.ALTERNATE_PRECEDENCE.value: dt_relation_trace.mp_alternate_precedence,
    ConstraintChecker.CHAIN_PRECEDENCE.value: dt_relation_trace.mp_chain_precedence,
    ConstraintChecker.NOT_RESPONDED_EXISTENCE.value: dt_negative_relation_trace.mp_not_responded_existence,
    ConstraintChecker.NOT_RESPONSE.value: dt_negative_relation_trace.mp_not_response,
    ConstraintChecker.NOT_CHAIN_RESPONSE.value: dt_negative_relation_trace.mp_not_chain_response,
    ConstraintChecker.NOT_PRECEDENCE.value: dt_negative_relation_trace.mp_not_precedence,
    ConstraintChecker.NOT_CHAIN_PRECEDENCE.value: dt_negative_relation_trace.mp_not_chain_precedence
}