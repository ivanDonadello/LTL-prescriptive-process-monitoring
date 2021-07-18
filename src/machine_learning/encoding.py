from src.constants import *
from src.machine_learning.labeling import *
from src.models.DTInput import *


def encode_traces(log, pairs, templates, rules, labeling):
    features = []
    encoded_data = []
    for trace in log:
        trace_result = {}
        for (a, b) in pairs:
            for template in templates:
                key = template.value + "[" + a + "," + b +"]"
                trace_result[key] = CONSTRAINT_CHECKER_FUNCTIONS[template.value](trace, True, a, b, rules).state.value
        if not features:
            features = list(trace_result.keys())
        encoded_data.append(list(trace_result.values()))
    labels = generate_labels(log, labeling)
    return DTInput(features, encoded_data, labels)
