from src.models.Prefix import *
from src.enums import PrefixType


def generate_prefixes(log, prefix_type):

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
        prefixes = {"UPTO": []}
        for index, trace in enumerate(log):
            print("Trace index: ", index)
            events = []
            for event in trace:
                events.append(event)
                prefix_model = Prefix(trace.attributes["concept:name"], index, events.copy())
                prefixes["UPTO"].append(prefix_model)
                if len(events) == n:
                    break
        return prefixes

    n = prefix_type["length"]
    if prefix_type["type"] == PrefixType.ONLY:
        return only(n)
    else:
        return up_to(n)


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

def generate_rules(rules):
    if rules.strip() == "":
        rules = "True"
        return rules
    if "is" in rules:
        rules = rules.replace("is", "==")
    words = rules.split()
    for index, word in enumerate(words):
        if "A." in word:
            words[index] = "A[\"" + word[2:] + "\"]"
            if not words[index + 2].isdigit():
                words[index + 2] = "\"" + words[index + 2] + "\""
        elif "T." in word:
            words[index] = "T[\"" + word[2:] + "\"]"
            if not words[index + 2].isdigit():
                words[index + 2] = "\"" + words[index + 2] + "\""
        elif word == "same":
            words[index] = "A[\"" + words[index + 1] + "\"] == T[\"" + words[index + 1] + "\"]"
            words[index + 1] = ""
    words = list(filter(lambda word: word != "", words))
    rules = " ".join(words)
    return rules
