from itertools import combinations
from collections import defaultdict


def get_num_traces_by_pairs(log, pairs):
    res = defaultdict(lambda: 0, {})
    for index, trace in enumerate(log):
        print("Trace index: ", index)
        for pair in pairs:
            a_exists = False
            b_exists = False
            for event in trace:
                if not a_exists and event["concept:name"] == pair[0]:
                    a_exists = True
                elif not b_exists and event["concept:name"] == pair[1]:
                    b_exists = True
                if a_exists and b_exists:
                    break
            if a_exists and b_exists:
                res[pair] += 1
    return res


# a-priori algorithm
# Description:
# pairs of events and their support (the % of traces where the pair of events occurs)
def a_priori(log):
    num_traces = len(log)
    distinct_events = set()
    print("Finding distinct events ...")
    for trace in log:
        for event in trace:
            distinct_events.add(event["concept:name"])
    print("Making event pairs ...")
    pairs = list(combinations(distinct_events, 2))
    print("Calculating frequency of pairs ...")
    result = get_num_traces_by_pairs(log, pairs)
    for key in result:
        result[key] /= num_traces
    return result


def find_pairs(log, support_threshold):
    frequent_pairs = [*{k: v for (k, v) in a_priori(log).items() if v > support_threshold}]
    pairs = []
    for pair in frequent_pairs:
        (x, y) = pair
        reverse_pair = (y, x)
        pairs.extend([pair, reverse_pair])
    return pairs