from declare_based.src.models.Prefix import *
from declare_based.src.enums import PrefixType


def generate_prefixes(log, prefix_type):
    def all(n):
        prefixes = {"ALL": []}
        for index, trace in enumerate(log):
            print("Trace index: ", index)
            events = []
            for event in trace:
                events.append(event)
                prefix_model = Prefix(trace.attributes["concept:name"], index, events.copy())
                prefixes["ALL"].append(prefix_model)
                if len(events) == n:
                    break
        return prefixes

    def only(n):
        prefixes = {n: []}
        for index, trace in enumerate(log):
            print("Trace index: ", index)
            if len(trace) >= n:
                events = []
                for event in trace:
                    events.append(event)
                    if len(events) == n:
                        prefix_model = Prefix(trace.attributes["concept:name"], index, events.copy())
                        prefixes[n].append(prefix_model)
                        break
        return prefixes

    def up_to(n):
        prefixes = {}
        for index, trace in enumerate(log):
            print("Trace index: ", index)
            events = []
            for event in trace:
                events.append(event)
                prefixModel = Prefix(trace.attributes["concept:name"], index, events.copy())
                key = len(prefixModel.events)
                if key not in prefixes:
                    prefixes[key] = []
                prefixes[key].append(prefixModel)
                if len(events) == n:
                    break
        return prefixes

    n = int(prefix_type["length"])
    if prefix_type["type"] == PrefixType.ONLY.value:
        return only(n)
    elif prefix_type["type"] == PrefixType.UPTO.value:
        return up_to(n)
    return all(n)


def parse_method(method):
    method_name = method.split("[")[0]
    rest = method.split("[")[1][:-1]
    if "," in rest:
        method_params = rest.split(",")
    else:
        method_params = [rest]
    return method_name, method_params


def generate_prefix_path(prefix):
    current_prefix = ""
    for event in prefix:
        current_prefix += event["concept:name"] + ", "
    current_prefix = current_prefix[:-2]
    return current_prefix
