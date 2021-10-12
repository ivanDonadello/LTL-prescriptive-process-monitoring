from src.enums.ConstraintChecker import ConstraintChecker
from src.constants.constants import CONSTRAINT_CHECKER_FUNCTIONS
from src.models.Prefix import *
from src.machine_learning.encoding import encode_traces
from src.machine_learning.apriori import generate_frequent_events_and_pairs
from src.machine_learning.decision_tree import generate_decision_tree, generate_boost_decision_tree
from src.enums import PrefixType
from sklearn.model_selection import train_test_split
import itertools
from src.enums import TraceLabel
import pandas as pd
import pdb


class ParamsOptimizer:
    def __init__(self, train_log, parameters, labeling, checkers, rules, train_ratio, val_ratio):
        self.parameter_names = parameters.keys()
        self.param_grid = [element for element in itertools.product(*parameters.values())]
        self.train_log = train_log
        self.parameters = parameters
        self.labeling = labeling
        self.checkers = checkers
        self.rules = rules
        self.model_grid = []
        self.real_val_ratio = val_ratio * len(self.train_log) / train_ratio

    def params_grid_search(self, dataset_name, constr_family):
        categories = [TraceLabel.FALSE.value, TraceLabel.TRUE.value]

        for param_id, param_tuple in enumerate(self.param_grid):
            model_dict = {'dataset_name': dataset_name, 'constr_family': constr_family, 'parameters': param_tuple,
                          'f1_score_val': None, 'f1_score_train': None, 'max_depth': 0, 'id': param_id, 'model': None,
                          'dt_input_features': None}

            (frequent_events, frequent_pairs) = generate_frequent_events_and_pairs(self.train_log, param_tuple[0])

            # Generating decision tree input
            dt_input = encode_traces(self.train_log, frequent_events=frequent_events, frequent_pairs=frequent_pairs,
                             checkers=self.checkers, rules=self.rules, labeling=self.labeling)

            X = pd.DataFrame(dt_input.encoded_data, columns=dt_input.features)
            y = pd.Categorical(dt_input.labels, categories=categories)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

            # Generating decision tree and its score on a validation set
            dtc, f1_score_val, f1_score_train = generate_decision_tree(X_train, X_val, y_train, y_val, class_weight=param_tuple[1], min_samples_split=param_tuple[2])
            model_dict['model'] = dtc
            model_dict['max_depth'] = dtc.tree_.max_depth
            model_dict['f1_score_val'] = f1_score_val
            model_dict['f1_score_train'] = f1_score_train
            model_dict['dt_input_features'] = dt_input.features
            self.model_grid.append(model_dict)

        sorted_models = sorted(self.model_grid, key=lambda d: d['f1_score'])
        best_model_dict = sorted_models[-1]
        return best_model_dict


def calcPathFitnessOnPrefix(prefix, path, rules):
    count = 0
    for rule in path.rules:
        template, rule_state = rule
        template_name, template_params = parse_method(template)

        result = None
        if template_name in [ConstraintChecker.EXISTENCE.value, ConstraintChecker.ABSENCE.value, ConstraintChecker.INIT.value, ConstraintChecker.EXACTLY.value]:
            result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](prefix, True, template_params[0], rules)
        else:
            result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](prefix, True, template_params[0], template_params[1], rules)

        if rule_state == result.state:
            count += 1
    return count / len(path.rules)

def generate_prefixes(log, prefixing):

    def only(n):
        prefixes = {n: []}
        for index, trace in enumerate(log):
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
            events = []
            for event in trace:
                events.append(event)
                prefix_model = Prefix(trace.attributes["concept:name"], index, events.copy())
                prefixes["UPTO"].append(prefix_model)
                if len(events) == n:
                    break
        return prefixes

    n = prefixing["length"]
    if prefixing["type"] == PrefixType.ONLY:
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
            words[index] = "A[\"" + words[index + 1] + \
                "\"] == T[\"" + words[index + 1] + "\"]"
            words[index + 1] = ""
    words = list(filter(lambda word: word != "", words))
    rules = " ".join(words)
    return rules
