from declare_based.src.enums import TraceLabel


def calc_mean_label_threshold(log, labeling):
    total = 0
    if labeling["labelType"] == "Trace duration":
        for trace in log:
            total += (trace[len(trace) - 1]["time:timestamp"] - trace[0]["time:timestamp"]).total_seconds()
    elif labeling["labelType"] == "Trace numerical attributes":
        trace_attribute = labeling["traceAttribute"]
        for trace in log:
            total += float(trace.attributes[trace_attribute])
    mean_label_threshold = total / len(log)
    return mean_label_threshold


def generate_label(trace, custom_label_threshold, labeling):
    if labeling["labelType"] == "Trace duration":
        time_diff = (
                trace[len(trace) - 1]["time:timestamp"] - trace[0]["time:timestamp"]
        ).total_seconds()
        if time_diff < custom_label_threshold:
            return TraceLabel.TRUE
        return TraceLabel.FALSE
    elif labeling["labelType"] == "Trace numerical attributes":
        trace_attribute = labeling["traceAttribute"]
        if float(trace.attributes[trace_attribute]) < custom_label_threshold:
            return TraceLabel.TRUE
        return TraceLabel.FALSE


def generate_labels(log, prefixes, custom_label_threshold, labeling):
    result = []
    for prefix in prefixes:
        result.append(generate_label(log[prefix.trace_num], custom_label_threshold, labeling).value)
    return result