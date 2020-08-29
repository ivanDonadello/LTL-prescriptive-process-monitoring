from declare_based.src.constants import *
from declare_based.src.machine_learning.labeling import *
from declare_based.src.models.DTInput import *


def encode_traces(log, pairs, templates, rules, labeling):
    features = []
    encoded_data = []
    for trace in log:
        trace_result = {}
        for (a, b) in pairs:
            for template in templates:
                trace_result[template + "[" + a + "," + b + "]"] = DT_TRACE_METHODS[template](trace, True,
                                                                                               a, b,
                                                                                               rules["activation"],
                                                                                               rules["correlation"],
                                                                                               rules["vacuousSatisfaction"]).state.value
        if not features:
            features = list(trace_result.keys())
        encoded_data.append(list(trace_result.values()))
    labels = generate_labels(log, labeling)
    return DTInput(features, encoded_data, labels)
