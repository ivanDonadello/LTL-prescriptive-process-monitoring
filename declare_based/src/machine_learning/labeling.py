from declare_based.src.enums import *


def calc_mean_label_threshold(log, labeling):
    total = 0
    if labeling["labelType"] == LabelType.TRACE_DURATION.value:
        for trace in log:
            total += (trace[len(trace) - 1]["time:timestamp"] - trace[0]["time:timestamp"]).total_seconds()
    elif labeling["labelType"] == LabelType.TRACE_NUMERICAL_ATTRIBUTES.value:
        trace_attribute = labeling["traceAttribute"]
        for trace in log:
            total += float(trace.attributes[trace_attribute])
    mean_label_threshold = total / len(log)
    return mean_label_threshold


def generate_label(trace, labeling):
    if labeling["labelType"] == LabelType.DEFAULT.value:
        if trace.attributes["label"] == "true":
            return TraceLabel.TRUE
        return TraceLabel.FALSE
    elif labeling["labelType"] == LabelType.TRACE_DURATION.value:
        time_diff = (
                trace[len(trace) - 1]["time:timestamp"] - trace[0]["time:timestamp"]
        ).total_seconds()
        if time_diff < labeling["customLabelThreshold"]:
            return TraceLabel.TRUE
        return TraceLabel.FALSE
    elif labeling["labelType"] == LabelType.TRACE_NUMERICAL_ATTRIBUTES.value:
        trace_attribute = labeling["traceAttribute"]
        if float(trace.attributes[trace_attribute]) < labeling["customLabelThreshold"]:
            return TraceLabel.TRUE
        return TraceLabel.FALSE


def generate_labels(log, prefixes, labeling):
    result = []
    for prefix in prefixes:
        result.append(generate_label(log[prefix.trace_num], labeling).value)
    return result