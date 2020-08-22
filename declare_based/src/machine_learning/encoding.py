from declare_based.src.constants import *
from declare_based.src.machine_learning.labeling import *
from declare_based.src.models.DTInput import *


def encode_prefixes(log, prefixes, pairs, templates, rules, custom_label_threshold, labeling):
    res = {}
    for key in prefixes:
        print("Prefix len: ", key)
        features = []
        encoded_data = []
        for prefix in prefixes[key]:
            print("Trace index of prefix: ", prefix.trace_num)
            prefix_result = {}
            for (a, b) in pairs:
                for template in templates:
                    prefix_result[template + "[" + a + "," + b + "]"] = DT_TRACE_METHODS[template](prefix.events, True,
                                                                                                   a, b,
                                                                                                   rules["activation"],
                                                                                                   rules["correlation"],
                                                                                                   rules["vacuousSatisfaction"]).state.value
            if not features:
                features = list(prefix_result.keys())
            encoded_data.append(list(prefix_result.values()))
        labels = generate_labels(log, prefixes[key], custom_label_threshold, labeling)
        if key not in res:
            res[key] = []
        res[key] = DTInput(len(prefixes[key]), features, encoded_data, labels)
    return res
