import pdb

from src.enums import *


def calc_mean_label_threshold(log, labeling):
    total = 0
    if labeling["type"] == LabelType.TRACE_DURATION:
        for trace in log:
            total += (trace[len(trace) - 1]["time:timestamp"] - trace[0]["time:timestamp"]).total_seconds()
    elif labeling["type"] == LabelType.TRACE_NUMERICAL_ATTRIBUTES:
        trace_attribute = labeling["trace_attribute"]
        for trace in log:
            total += float(trace.attributes[trace_attribute])
    mean_label_threshold = total / len(log)
    return mean_label_threshold


def generate_label(trace, labeling):
    if labeling["type"] == LabelType.DEFAULT:
        if trace.attributes["label"] == "true":
            return TraceLabel.TRUE
        return TraceLabel.FALSE
    elif labeling["type"] == LabelType.TRACE_CATEGORICAL_ATTRIBUTES:
        if trace[0][labeling["trace_lbl_attr"]] == labeling["trace_label"]:
            return TraceLabel.TRUE
        return TraceLabel.FALSE
    elif labeling["type"] == LabelType.TRACE_DURATION:
        time_diff = (
                trace[len(trace) - 1]["time:timestamp"] - trace[0]["time:timestamp"]
        ).total_seconds()
        if time_diff < labeling["custom_threshold"]:
            return TraceLabel.TRUE
        return TraceLabel.FALSE
    elif labeling["type"] == LabelType.TRACE_NUMERICAL_ATTRIBUTES:
        trace_attribute = labeling["trace_attribute"]
        if float(trace.attributes[trace_attribute]) < labeling["custom_threshold"]:
            return TraceLabel.TRUE
        return TraceLabel.FALSE


def generate_labels(log, labeling):
    result = []
    for trace in log:
        result.append(generate_label(trace, labeling).value)
    return result
