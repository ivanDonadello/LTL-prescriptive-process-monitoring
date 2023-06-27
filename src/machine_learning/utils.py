from src.enums.ConstraintChecker import ConstraintChecker
from src.enums import TraceState
from src.constants.constants import CONSTRAINT_CHECKER_FUNCTIONS
from src.models.Prefix import *
from src.machine_learning.encoding import encode_traces
from src.machine_learning.apriori import generate_frequent_events_and_pairs
# from src.machine_learning.decision_tree import generate_decision_tree, generate_paths, generate_boost_decision_tree
from src.enums import PrefixType
from sklearn.model_selection import train_test_split
import itertools
from src.enums import TraceLabel
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
import pdb
import settings
import math


def gain(c, nc, pc, pnc):
    prob_pos_comp = (pc + settings.smooth_factor) / (c + settings.smooth_factor * settings.num_classes)
    prob_pos_non_comp = (pnc + settings.smooth_factor) / (nc + settings.smooth_factor * settings.num_classes)
    _gain = prob_pos_comp / prob_pos_non_comp
    return _gain


def matthews_corrcoef(tp, fp, fn, tn):
    num = tp*tn - fp*fn
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if denom == 0:
        return 0
    else:
        return num/denom


def calcScore(path, pos_paths_total_samples_, weights):
    purity = 1 - path.impurity
    pos_probabiity = path.num_samples['positive']/pos_paths_total_samples_
    w = np.array([0.8, 0.1, 0.1])
    w = np.array([0.0, 0, 0])
    # pdb.set_trace()

    if path.num_samples['node_samples'] > 2:
        w = weights
    # pdb.set_trace()
    return np.mean(w*np.array([path.fitness, purity, pos_probabiity]))
    # return path.fitness*1#pos_probabiity


def calcPathFitnessOnPrefixGOOD(prefix, path, rules, fitness_type):
    path_weights = []
    path_activated_rules = np.zeros(len(path.rules))
    fitness = None
    for rule_idx, rule in enumerate(path.rules):
        template, rule_state, _ = rule
        template_name, template_params = parse_method(template)

        result = None
        if template_name in [ConstraintChecker.EXISTENCE.value, ConstraintChecker.ABSENCE.value, ConstraintChecker.INIT.value, ConstraintChecker.EXACTLY.value]:
            result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](prefix, True, template_params[0], rules)
        else:
            result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](prefix, True, template_params[0], template_params[1], rules)

        if rule_state == result.state:
            path_activated_rules[rule_idx] = 1
        path_weights.append(1/(rule_idx+1))

    if fitness_type == 'mean':
        fitness = np.mean(path_activated_rules)
    elif fitness_type == 'wmean':
        fitness = np.sum(path_weights*path_activated_rules)/np.sum(path_weights)

    return fitness


def calcPathFitnessOnPrefix(prefix, path, rules, fitness_type):
    path_weights = []
    path_activated_rules = np.zeros(len(path.rules))
    fitness = None
    for rule_idx, rule in enumerate(path.rules):
        template, rule_state, _ = rule
        template_name, template_params = parse_method(template)

        result = None
        if settings.use_score:
            if template_name in [ConstraintChecker.EXISTENCE.value, ConstraintChecker.ABSENCE.value, ConstraintChecker.INIT.value, ConstraintChecker.EXACTLY.value]:
                result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](prefix, False, template_params[0], rules)
            else:
                result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](prefix, False, template_params[0], template_params[1], rules)

            if rule_state == TraceState.VIOLATED and result.state == TraceState.POSSIBLY_VIOLATED:
                path_activated_rules[rule_idx] = 1
            elif rule_state == TraceState.SATISFIED and result.state == TraceState.POSSIBLY_VIOLATED:
                path_activated_rules[rule_idx] = 0.5
            elif rule_state == TraceState.VIOLATED and result.state == TraceState.POSSIBLY_SATISFIED:
                path_activated_rules[rule_idx] = 0.5
            elif rule_state == TraceState.SATISFIED and result.state == TraceState.POSSIBLY_SATISFIED:
                path_activated_rules[rule_idx] = 1
        else:
            if template_name in [ConstraintChecker.EXISTENCE.value, ConstraintChecker.ABSENCE.value, ConstraintChecker.INIT.value, ConstraintChecker.EXACTLY.value]:
                result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](prefix, True, template_params[0], rules)
            else:
                result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](prefix, True, template_params[0], template_params[1], rules)

        if rule_state == result.state:
            path_activated_rules[rule_idx] = 1
        path_weights.append(1/(rule_idx+1))

    if fitness_type == 'mean':
        fitness = np.mean(path_activated_rules)
    elif fitness_type == 'wmean':
        fitness = np.sum(path_weights*path_activated_rules)/np.sum(path_weights)

    return fitness


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
