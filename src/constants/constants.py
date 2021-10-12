from src.enums.ConstraintChecker import ConstraintChecker
from src.constraint_checkers import choice
from src.constraint_checkers import existence
from src.constraint_checkers import relation
from src.constraint_checkers import negative_relation

CONSTRAINT_CHECKER_FUNCTIONS = {
    ConstraintChecker.EXISTENCE.value: existence.mp_existence,
    ConstraintChecker.ABSENCE.value: existence.mp_absence,
    ConstraintChecker.INIT.value: existence.mp_init,
    ConstraintChecker.EXACTLY.value: existence.mp_exactly,
    ConstraintChecker.CHOICE.value: choice.mp_choice,
    ConstraintChecker.EXCLUSIVE_CHOICE.value: choice.mp_exclusive_choice,
    ConstraintChecker.RESPONDED_EXISTENCE.value: relation.mp_responded_existence,
    ConstraintChecker.RESPONSE.value: relation.mp_response,
    ConstraintChecker.ALTERNATE_RESPONSE.value: relation.mp_alternate_response,
    ConstraintChecker.CHAIN_RESPONSE.value: relation.mp_chain_response,
    ConstraintChecker.PRECEDENCE.value: relation.mp_precedence,
    ConstraintChecker.ALTERNATE_PRECEDENCE.value: relation.mp_alternate_precedence,
    ConstraintChecker.CHAIN_PRECEDENCE.value: relation.mp_chain_precedence,
    ConstraintChecker.NOT_RESPONDED_EXISTENCE.value: negative_relation.mp_not_responded_existence,
    ConstraintChecker.NOT_RESPONSE.value: negative_relation.mp_not_response,
    ConstraintChecker.NOT_CHAIN_RESPONSE.value: negative_relation.mp_not_chain_response,
    ConstraintChecker.NOT_PRECEDENCE.value: negative_relation.mp_not_precedence,
    ConstraintChecker.NOT_CHAIN_PRECEDENCE.value: negative_relation.mp_not_chain_precedence
}