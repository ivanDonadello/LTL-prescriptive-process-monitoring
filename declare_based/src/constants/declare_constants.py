from declare_based.src.enums.ConstraintChecker import ConstraintChecker
from declare_based.src.declare_templates.log import existence as dt_existence_log
from declare_based.src.declare_templates.log import choice as dt_choice_log
from declare_based.src.declare_templates.log import relation as dt_relation_log
from declare_based.src.declare_templates.log import negative_relation as dt_negative_relation_log
from declare_based.src.declare_templates.trace import relation as dt_relation_trace
from declare_based.src.declare_templates.trace import negative_relation as dt_negative_relation_trace

DT_LOG_METHODS = {
    ConstraintChecker.INIT.value: dt_existence_log.mp_init,
    ConstraintChecker.EXISTENCE.value: dt_existence_log.mp_existence,
    ConstraintChecker.ABSENCE.value: dt_existence_log.mp_absence,
    ConstraintChecker.EXACTLY.value: dt_existence_log.mp_exactly,
    ConstraintChecker.CHOICE.value: dt_choice_log.mp_choice,
    ConstraintChecker.EXCLUSIVE_CHOICE.value: dt_choice_log.mp_exclusive_choice,
    ConstraintChecker.RESPONDED_EXISTENCE.value: dt_relation_log.mp_responded_existence,
    ConstraintChecker.RESPONSE.value: dt_relation_log.mp_response,
    ConstraintChecker.ALTERNATE_RESPONSE.value: dt_relation_log.mp_alternate_response,
    ConstraintChecker.CHAIN_RESPONSE.value: dt_relation_log.mp_chain_response,
    ConstraintChecker.PRECEDENCE.value: dt_relation_log.mp_precedence,
    ConstraintChecker.ALTERNATE_PRECEDENCE.value: dt_relation_log.mp_alternate_precedence,
    ConstraintChecker.CHAIN_PRECEDENCE.value: dt_relation_log.mp_chain_precedence,
    ConstraintChecker.NOT_RESPONDED_EXISTENCE.value: dt_negative_relation_log.mp_not_responded_existence,
    ConstraintChecker.NOT_RESPONSE.value: dt_negative_relation_log.mp_not_response,
    ConstraintChecker.NOT_CHAIN_RESPONSE.value: dt_negative_relation_log.mp_not_chain_response,
    ConstraintChecker.NOT_PRECEDENCE.value: dt_negative_relation_log.mp_not_precedence,
    ConstraintChecker.NOT_CHAIN_PRECEDENCE.value: dt_negative_relation_log.mp_not_chain_precedence
}

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