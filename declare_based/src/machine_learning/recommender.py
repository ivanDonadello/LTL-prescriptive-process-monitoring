from declare_based.src.machine_learning.utils import *
from declare_based.src.machine_learning.apriori import *
from declare_based.src.machine_learning.encoding import *
from declare_based.src.machine_learning.decision_tree import *
from declare_based.src.models import EvaluationResult
from declare_based.src.constants import *

import numpy as np
from sklearn import metrics

import csv


def recommend(prefix, path, rules):
    recommendation = ""
    for rule in path.rules:
        method, state = rule
        method_name, method_params = parse_method(method)
        result = DT_TRACE_METHODS[method_name](prefix, False, method_params[0], method_params[1], rules["activation"],
                                               rules["correlation"], rules["vacuousSatisfaction"])
        if state == TraceState.SATISFIED:
            if result.state == TraceState.VIOLATED:
                recommendation = "Contradiction"
                break
            elif result.state == TraceState.SATISFIED:
                pass
            elif result.state == TraceState.POSSIBLY_VIOLATED:
                recommendation += method + " should be SATISFIED. "
            elif result.state == TraceState.POSSIBLY_SATISFIED:
                recommendation += method + " should not be VIOLATED. "
        elif state == TraceState.VIOLATED:
            if result.state == TraceState.VIOLATED:
                pass
            elif result.state == TraceState.SATISFIED:
                recommendation = "Contradiction"
                break
            elif result.state == TraceState.POSSIBLY_VIOLATED:
                recommendation += method + " should not be SATISFIED. "
            elif result.state == TraceState.POSSIBLY_SATISFIED:
                recommendation += method + " should be VIOLATED. "
    return recommendation


def evaluate(trace, prefix, target_label, path, rules, labeling):
    is_compliant = True
    for rule in path.rules:
        method, state = rule
        method_name, method_params = parse_method(method)
        result = DT_TRACE_METHODS[method_name](prefix.events, True, method_params[0], method_params[1],
                                               rules["activation"], rules["correlation"], rules["vacuousSatisfaction"])
        if state != result.state:
            is_compliant = False
            break

    label = generate_label(trace, labeling)

    if target_label == TraceLabel.TRUE.name:
        if is_compliant:
            cm = ConfusionMatrix.TP if label == TraceLabel.TRUE else ConfusionMatrix.FP
        else:
            cm = ConfusionMatrix.FN if label == TraceLabel.TRUE else ConfusionMatrix.TN
    else:
        if is_compliant:
            cm = ConfusionMatrix.FN if label == TraceLabel.TRUE else ConfusionMatrix.TN
        else:
            cm = ConfusionMatrix.TP if label == TraceLabel.TRUE else ConfusionMatrix.FP
    return is_compliant, cm


def generate_recommendations_and_evaluation(test_log, train_log, labeling, prefix_type, support_threshold, templates,
                                            rules):
    if rules["vacuousSatisfaction"] == "TRUE":
        rules["vacuousSatisfaction"] = True
    else:
        rules["vacuousSatisfaction"] = False

    if labeling["labelThresholdType"] == LabelThresholdType.LABEL_MEAN.value:
        labeling["customLabelThreshold"] = calc_mean_label_threshold(train_log, labeling)
    elif labeling["labelThresholdType"] == LabelThresholdType.CUSTOM.value:
        labeling["customLabelThreshold"] = float(labeling["customLabelThreshold"])
    else:
        labeling["customLabelThreshold"] = None

    target_label = labeling["targetLabel"]

    pairs = find_pairs(train_log, support_threshold)

    print("Generating train prefixes ...")
    train_prefixes = generate_prefixes(train_log, prefix_type)

    print("Generating test prefixes ...")
    test_prefixes = generate_prefixes(test_log, prefix_type)

    print("Generating decision tree input ...")
    dt_input = encode_prefixes(train_log, prefixes=train_prefixes, pairs=pairs, templates=templates, rules=rules, labeling=labeling)

    print("Generating paths ...")
    paths = generate_decision_tree_paths(dt_input=dt_input, target_label=target_label)

    print("Generating recommendations ...")
    recommendations = []
    eval_res = EvaluationResult()
    y = []
    pred = []
    for key in test_prefixes:
        for prefix in test_prefixes[key]:
            if key in paths:
                for path in paths[key]:
                    recommendation = recommend(prefix.events, path, rules)
                    if recommendation != "Contradiction":
                        trace = test_log[prefix.trace_num]
                        is_compliant, e = evaluate(trace, prefix, target_label, path, rules, labeling)
                        if e == ConfusionMatrix.TP:
                            eval_res.tp += 1
                        elif e == ConfusionMatrix.FP:
                            eval_res.fp += 1
                        elif e == ConfusionMatrix.FN:
                            eval_res.fn += 1
                        elif e == ConfusionMatrix.TN:
                            eval_res.tn += 1

                        recommendation_model = Recommendation(
                            trace_id=prefix.trace_id,
                            prefix_len=len(prefix.events),
                            complete_trace=generate_prefix_path(test_log[prefix.trace_num]),
                            current_prefix=generate_prefix_path(prefix.events),
                            actual_label=generate_label(trace, labeling).name,
                            target_label=target_label,
                            is_compliant=str(is_compliant).upper(),
                            confusion_matrix=e.name,
                            impurity=path.impurity,
                            num_samples = path.num_samples,
                            recommendation=recommendation
                        )
                        y.append(recommendation_model.actual_label)
                        pred.append(recommendation_model.num_samples["positive"] / recommendation_model.num_samples["total"])
                        recommendations.append(recommendation_model)
                        break
    try:
        eval_res.precision = eval_res.tp / (eval_res.tp + eval_res.fp)
    except ZeroDivisionError:
        eval_res.precision = 0

    try:
        eval_res.recall = eval_res.tp / (eval_res.tp + eval_res.fn)
    except ZeroDivisionError:
        eval_res.recall = 0

    try:
        eval_res.accuracy = (eval_res.tp + eval_res.tn) / (eval_res.tp + eval_res.fp + eval_res.fn + eval_res.tn)
    except ZeroDivisionError:
        eval_res.accuracy = 0

    try:
        eval_res.fscore = 2 * eval_res.precision * eval_res.recall / (eval_res.precision + eval_res.recall)
    except ZeroDivisionError:
        eval_res.fscore = 0

    fpr, tpr, thresholds = metrics.roc_curve(np.array(y), np.array(pred), pos_label=target_label)
    eval_res.auc = metrics.auc(fpr, tpr)

    print("Writing evaluation result into csv file")
    write_evaluation_to_csv(eval_res)

    print("Writing recommendations into csv file")
    write_recommendations_to_csv(recommendations)

    return recommendations, eval_res


def write_evaluation_to_csv(e):
    os.makedirs(os.path.join(settings.MEDIA_ROOT + "output/result"))
    csv_file = settings.MEDIA_ROOT + "output/result/evaluation.csv"
    fieldnames = ["tp", "fp", "tn", "fn", "precision", "recall", "accuracy", "fscore", "auc"]
    values = {
        "tp": e.tp,
        "fp": e.fp,
        "tn": e.tn,
        "fn": e.fn,
        "precision": e.precision,
        "recall": e.recall,
        "accuracy": e.accuracy,
        "fscore": e.fscore,
        "auc": e.auc
    }
    try:
        with open(csv_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(values)
    except IOError:
        print("I/O error")


def write_recommendations_to_csv(recommendations):
    csv_file = settings.MEDIA_ROOT + "output/result/recommendations.csv"
    fieldnames = ["Trace id", "Prefix len", "Complete trace", "Current prefix", "Recommendation", "Actual label", "Target label", "Compliant", "Confusion matrix", "Impurity", "Num samples"]
    values = []
    for r in recommendations:
        values.append(
            {
                "Trace id": r.trace_id,
                "Prefix len": r.prefix_len,
                "Complete trace": r.complete_trace,
                "Current prefix": r.current_prefix,
                "Recommendation": r.recommendation,
                "Actual label": r.actual_label,
                "Target label": r.target_label,
                "Compliant": r.is_compliant,
                "Confusion matrix": r.confusion_matrix,
                "Impurity": r.impurity,
                "Num samples": r.num_samples["total"]
            }
        )

    try:
        with open(csv_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for value in values:
                writer.writerow(value)
    except IOError:
        print("I/O error")