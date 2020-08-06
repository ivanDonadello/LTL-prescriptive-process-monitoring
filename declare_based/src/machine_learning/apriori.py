from itertools import combinations


def get_num_traces_by_two_events(log, a, b):
    num_traces_satisfied = 0
    for trace in log:
        a_exists = False
        b_exists = False
        for event in trace:
            if not a_exists and event["concept:name"] == a:
                a_exists = True
            elif not b_exists and event["concept:name"] == b:
                b_exists = True
            if a_exists and b_exists:
                break
        if a_exists and b_exists:
            num_traces_satisfied += 1
    return num_traces_satisfied


# a-priori algorithm
# Description:
# pairs of events and their support (the % of traces where the pair of events occurs)
def a_priori(log):
    num_traces = len(log)
    distinct_events = set()
    result = {}
    for trace in log:
        for event in trace:
            distinct_events.add(event["concept:name"])
    pairs = list(combinations(distinct_events, 2))
    for pair in pairs:
        print(pair)
        result[pair] = get_num_traces_by_two_events(log, pair[0], pair[1]) / num_traces
    return result


def find_pairs(log, support_threshold):
    frequent_pairs = [*{k: v for (k, v) in a_priori(log).items() if v > support_threshold}]
    pairs = []
    for pair in frequent_pairs:
        (x, y) = pair
        reverse_pair = (y, x)
        pairs.extend([pair, reverse_pair])
    return pairs