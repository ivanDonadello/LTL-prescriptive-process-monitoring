from src.constants import *
from src.machine_learning.labeling import *
from src.models.DTInput import *
from src.enums.ConstraintChecker import *
import settings


def encode_traces(log, frequent_events, frequent_pairs, checkers, rules, labeling):
    event_checkers = list(filter(lambda checker: checker in [ConstraintChecker.EXISTENCE, ConstraintChecker.ABSENCE, ConstraintChecker.INIT, ConstraintChecker.EXACTLY], checkers))
    pair_checkers = list(filter(lambda checker: checker not in [ConstraintChecker.EXISTENCE, ConstraintChecker.ABSENCE, ConstraintChecker.INIT, ConstraintChecker.EXACTLY], checkers))

    features = []
    encoded_data = []

    for trace in log:
        trace_result = {}
        for a in frequent_events:
            for checker in event_checkers:
                key = checker.value + "[" + a + "]"
                trace_result[key] = CONSTRAINT_CHECKER_FUNCTIONS[checker.value](trace, settings.one_hot_encoding, a, rules).state.value
        for (a, b) in frequent_pairs:
            for checker in pair_checkers:
                key = checker.value + "[" + a + "," + b +"]"
                trace_result[key] = CONSTRAINT_CHECKER_FUNCTIONS[checker.value](trace, settings.one_hot_encoding, a, b, rules).state.value
        if not features:
            features = list(trace_result.keys())
        encoded_data.append(list(trace_result.values()))
    labels = generate_labels(log, labeling)
    return DTInput(features, encoded_data, labels)
