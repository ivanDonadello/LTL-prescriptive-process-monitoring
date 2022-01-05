from itertools import combinations
from collections import defaultdict


def get_frequent_events(log, support_threshold):
    res = defaultdict(lambda: 0, {})
    for trace in log:
        for event in trace:
            event_name = event["concept:name"]
            res[event_name] += 1
    frequent_events = []
    for key in res:
        if res[key] / len(log) > support_threshold:
            frequent_events.append(key)
    return frequent_events


def get_frequent_pairs(log, pairs, support_threshold):
    res = defaultdict(lambda: 0, {})
    for trace in log:
        for pair in pairs:
            a_exists = False
            b_exists = False
            for event in trace:
                if not a_exists and event["concept:name"] == pair[0]:
                    a_exists = True
                elif not b_exists and event["concept:name"] == pair[1]:
                    b_exists = True
                if a_exists and b_exists:
                    res[pair] += 1
                    break
    frequent_pairs = []
    for key in res:
        if res[key] / len(log) > support_threshold:
            frequent_pairs.append(key)
    return frequent_pairs


def generate_frequent_events_and_pairs(log, support_threshold):
    print("Finding frequent events ...")
    frequent_events = get_frequent_events(log, support_threshold)

    print("Making event pairs ...")
    pairs = list(combinations(frequent_events, 2))

    print("Finding frequent pairs ...")
    frequent_pairs = get_frequent_pairs(log, pairs, support_threshold)

    all_frequent_pairs = []
    for pair in frequent_pairs:
        (x, y) = pair
        reverse_pair = (y, x)
        all_frequent_pairs.extend([pair, reverse_pair])
    return frequent_events, all_frequent_pairs
