import pdb

from src.machine_learning.utils import *
from src.machine_learning.apriori import *
from src.machine_learning.encoding import *
from src.machine_learning.decision_tree import *
from src.models import EvaluationResult
from src.constants import *
import csv
import numpy as np
from sklearn import metrics


class ParamsOptimizer:
    def __init__(self, train_val_log, val_log, train_log, parameters, labeling, checkers, rules, min_prefix_length, max_prefix_length):
        self.parameter_names = parameters.keys()
        self.val_log = val_log
        self.train_val_log = train_val_log
        self.param_grid = [element for element in itertools.product(*parameters.values())]
        self.train_log = train_log
        self.parameters = parameters
        self.labeling = labeling
        self.checkers = checkers
        self.rules = rules
        self.model_grid = []
        self.min_prefix_length = min_prefix_length
        self.max_prefix_length = max_prefix_length

    def params_grid_search(self, dataset_name, constr_family):
        categories = [TraceLabel.FALSE.value, TraceLabel.TRUE.value]

        for param_id, param_tuple in enumerate(self.param_grid):
            model_dict = {'dataset_name': dataset_name, 'constr_family': constr_family, 'parameters': param_tuple,
                          'f1_score_val': None, 'f1_score_train': None, 'f1_prefix_val': None, 'max_depth': 0,
                          'id': param_id, 'model': None, 'bo':None}

            (frequent_events_train, frequent_pairs_train) = generate_frequent_events_and_pairs(self.train_val_log, param_tuple[0])

            # Generating decision tree input
            dt_input_train = encode_traces(self.train_log, frequent_events=frequent_events_train,
                                           frequent_pairs=frequent_pairs_train, checkers=self.checkers,
                                           rules=self.rules, labeling=self.labeling)
            dt_input_val = encode_traces(self.val_log, frequent_events=frequent_events_train,
                                         frequent_pairs=frequent_pairs_train, checkers=self.checkers,
                                         rules=self.rules, labeling=self.labeling)

            X_train = pd.DataFrame(dt_input_train.encoded_data, columns=dt_input_train.features)
            y_train = pd.Categorical(dt_input_train.labels, categories=categories)

            X_val = pd.DataFrame(dt_input_val.encoded_data, columns=dt_input_val.features)
            y_val = pd.Categorical(dt_input_val.labels, categories=categories)

            # Generating decision tree and its score on a validation set
            dtc, f1_score_val, f1_score_train = generate_decision_tree(X_train, X_val, y_train, y_val, class_weight=param_tuple[1], min_samples_split=param_tuple[2])
            print(param_tuple)
            paths = generate_paths(dtc=dtc, dt_input_features=dt_input_train.features, target_label=self.labeling["target"])

            # Evaluation on val set prefixes
            results = []
            for pref_id, prefix_len in enumerate(range(self.min_prefix_length, self.max_prefix_length + 1)):
                prefixing = {
                    "type": PrefixType.ONLY,
                    "length": prefix_len
                }

                evaluation = evaluate_recommendations(input_log=self.val_log,
                                                                       labeling=self.labeling, prefixing=prefixing,
                                                                       rules=self.rules, paths=paths)
                results.append(evaluation)

            model_dict['model'] = dtc
            model_dict['max_depth'] = dtc.tree_.max_depth
            model_dict['f1_score_val'] = f1_score_val
            model_dict['f1_score_train'] = f1_score_train
            model_dict['f1_prefix_val'] = np.average([res.fscore for res in results])
            self.model_grid.append(model_dict)

        # retrain the DT using trainval set with the best params trested on the val set
        sorted_models = sorted(self.model_grid, key=lambda d: d['f1_prefix_val'])
        best_model_dict = sorted_models[-1]
        #pdb.set_trace()
        (frequent_events_trainval, frequent_pairs_trainval) = generate_frequent_events_and_pairs(self.train_val_log,
                                                                                           best_model_dict['parameters'][0])
        dt_input_trainval = encode_traces(self.train_val_log, frequent_events=frequent_events_trainval,
                                          frequent_pairs=frequent_pairs_trainval, checkers=self.checkers,
                                          rules=self.rules, labeling=self.labeling)
        dt_input_val = encode_traces(self.val_log, frequent_events=frequent_events_trainval,
                                     frequent_pairs=frequent_pairs_trainval, checkers=self.checkers,
                                     rules=self.rules, labeling=self.labeling)

        X_train_val = pd.DataFrame(dt_input_trainval.encoded_data, columns=dt_input_trainval.features)
        y_train_val = pd.Categorical(dt_input_trainval.labels, categories=categories)
        X_val = pd.DataFrame(dt_input_val.encoded_data, columns=dt_input_val.features)
        y_val = pd.Categorical(dt_input_val.labels, categories=categories)

        dtc, _, _ = generate_decision_tree(X_train_val, X_val, y_train_val, y_val,
                                                                   class_weight=best_model_dict['parameters'][1],
                                                                   min_samples_split=best_model_dict['parameters'][2])
        best_model_dict['model'] = dtc
        return best_model_dict, dt_input_trainval.features

    def params_grid_search_old(self, dataset_name, constr_family):
        categories = [TraceLabel.FALSE.value, TraceLabel.TRUE.value]

        for param_id, param_tuple in enumerate(self.param_grid):
            model_dict = {'dataset_name': dataset_name, 'constr_family': constr_family, 'parameters': param_tuple,
                          'f1_score_val': None, 'f1_score_train': None, 'max_depth': 0, 'id': param_id, 'model': None,
                          'dt_input_features': None}

            (frequent_events, frequent_pairs) = generate_frequent_events_and_pairs(self.train_log, param_tuple[0])

            # Generating decision tree input
            dt_input = encode_traces(log=self.train_log, frequent_events=frequent_events, frequent_pairs=frequent_pairs,
                                     checkers=self.checkers, rules=self.rules, labeling=self.labeling)

            X = pd.DataFrame(dt_input.encoded_data, columns=dt_input.features)
            y = pd.Categorical(dt_input.labels, categories=categories)
            #X_new = SelectKBest(mutual_info_classif, k=int(0.7*X.shape[1])).fit_transform(X, y)

            # Tree
            #clf = ExtraTreesClassifier(n_estimators=50)
            #clf = clf.fit(X, y)
            #model = SelectFromModel(clf, prefit=True)
            #X_new = model.transform(X)

            X_new = X
            # Create the RFE object and compute a cross-validated score.
            #svc = SVC(kernel="linear")
            #rfe = RFE(estimator=svc, step=1)
            #rfe.fit(X, y)
            #X_new = rfe.transform(X)

            print(X_new.shape)

            X_train, X_val, y_train, y_val = train_test_split(X_new, y, test_size=0.1, random_state=42)

            # Generating decision tree and its score on a validation set
            dtc, f1_score_val, f1_score_train = generate_decision_tree(X_train, X_val, y_train, y_val,
                                                                       class_weight=param_tuple[1],
                                                                       min_samples_split=param_tuple[2])
            model_dict['model'] = dtc
            model_dict['max_depth'] = dtc.tree_.max_depth
            model_dict['f1_score_val'] = f1_score_val
            model_dict['f1_score_train'] = f1_score_train
            model_dict['dt_input_features'] = dt_input.features
            self.model_grid.append(model_dict)

        sorted_models = sorted(self.model_grid, key=lambda d: d['f1_score_val'])
        best_model_dict = sorted_models[-1]
        return best_model_dict


def recommend(prefix, path, rules):
    recommendation = ""
    for rule in path.rules:
        template, rule_state = rule
        template_name, template_params = parse_method(template)

        result = None
        if template_name in [ConstraintChecker.EXISTENCE.value, ConstraintChecker.ABSENCE.value, ConstraintChecker.INIT.value, ConstraintChecker.EXACTLY.value]:
            result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](prefix, False, template_params[0], rules)
        else:
            result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](prefix, False, template_params[0], template_params[1], rules)

        if rule_state == TraceState.SATISFIED:
            if result.state == TraceState.VIOLATED:
                recommendation = "Contradiction"
                break
            elif result.state == TraceState.SATISFIED:
                pass
            elif result.state == TraceState.POSSIBLY_VIOLATED:
                recommendation += template + " should be SATISFIED. "
            elif result.state == TraceState.POSSIBLY_SATISFIED:
                recommendation += template + " should not be VIOLATED. "
        elif rule_state == TraceState.VIOLATED:
            if result.state == TraceState.VIOLATED:
                pass
            elif result.state == TraceState.SATISFIED:
                recommendation = "Contradiction"
                break
            elif result.state == TraceState.POSSIBLY_VIOLATED:
                recommendation += template + " should not be SATISFIED. "
            elif result.state == TraceState.POSSIBLY_SATISFIED:
                recommendation += template + " should be VIOLATED. "
    return recommendation


def evaluate(trace, path, rules, labeling):
    is_compliant = True
    for rule in path.rules:
        template, rule_state = rule
        template_name, template_params = parse_method(template)

        result = None
        if template_name in [ConstraintChecker.EXISTENCE.value, ConstraintChecker.ABSENCE.value, ConstraintChecker.INIT.value, ConstraintChecker.EXACTLY.value]:
            result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](trace, True, template_params[0], rules)
        else:
            result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](trace, True, template_params[0], template_params[1], rules)

        # if traccia compliant with path
        if rule_state != result.state:
            is_compliant = False
            break

    label = generate_label(trace, labeling)

    if labeling["target"] == TraceLabel.TRUE:
        if is_compliant:
            cm = ConfusionMatrix.TP if label == TraceLabel.TRUE else ConfusionMatrix.FP
            #print(f"1,{label},{cm}")
        else:
            cm = ConfusionMatrix.FN if label == TraceLabel.TRUE else ConfusionMatrix.TN
            #print(f"0,{label},{cm}")
    else:
        print("---------------------")
        if is_compliant:
            cm = ConfusionMatrix.FN if label == TraceLabel.TRUE else ConfusionMatrix.TN
        else:
            cm = ConfusionMatrix.TP if label == TraceLabel.TRUE else ConfusionMatrix.FP
    return is_compliant, cm


def test_dt(test_log, train_log, labeling, prefixing, support_threshold, checkers, rules):

    (frequent_events, frequent_pairs) = generate_frequent_events_and_pairs(train_log, support_threshold)

    print("Generating decision tree input ...")
    dt_input = encode_traces(train_log, frequent_events=frequent_events, frequent_pairs=frequent_pairs,
                             checkers=checkers, rules=rules, labeling=labeling)

    print("Generating decision tree ...")
    return dt_score(dt_input=dt_input)


def train_path_recommender(train_val_log, val_log, train_log, labeling, support_threshold, checkers, rules,
                           dataset_name, constr_family, output_dir, min_prefix_length, max_prefix_length):
    if labeling["threshold_type"] == LabelThresholdType.LABEL_MEAN:
        labeling["custom_threshold"] = calc_mean_label_threshold(train_log, labeling)

    target_label = labeling["target"]

    parameters = {'support_threshold': [support_threshold-0.2, support_threshold-0.1, support_threshold, support_threshold+0.1],
                  'class_weight': [None, 'balanced'],
                  'min_samples_split': [2]}
    """
    parameters = {'support_threshold': [support_threshold],
                 'class_weight': [None],
                  'min_samples_split': [2]}
    """

    if dataset_name == 'traffic_fines_1':
        parameters['class_weight'] = [None]

    print("Generating decision tree with params optimization ...")
    param_opt = ParamsOptimizer(train_val_log, val_log, train_log, parameters, labeling, checkers, rules, min_prefix_length,
                                max_prefix_length)
    best_model_dict, feature_names  = param_opt.params_grid_search(dataset_name, constr_family)

    with open(os.path.join(output_dir, 'model_params.csv'), 'a') as f:
        w = csv.writer(f)
        w.writerow(best_model_dict.values())

    print("Generating decision tree paths ...")
    paths = generate_paths(dtc=best_model_dict['model'], dt_input_features=feature_names,
                           target_label=target_label)
    return paths


def evaluate_recommendations(input_log, labeling, prefixing, rules, paths):
    #if labeling["threshold_type"] == LabelThresholdType.LABEL_MEAN:
    #    labeling["custom_threshold"] = calc_mean_label_threshold(train_log, labeling)

    target_label = labeling["target"]

    print("Generating test prefixes ...")
    prefixes = generate_prefixes(input_log, prefixing)

    eval_res = EvaluationResult()

    for prefix_length in prefixes:
        for prefix in prefixes[prefix_length]:
            for path in paths:
                path.fitness = calcPathFitnessOnPrefix(prefix.events, path, rules)

            paths = sorted(paths, key=lambda path: (- path.fitness, path.impurity, - path.num_samples["total"]),
                           reverse=False)

            selected_path = None

            for path_index, path in enumerate(paths):

                if selected_path and (
                        path.fitness != selected_path.fitness or path.impurity != selected_path.impurity or path.num_samples != selected_path.num_samples):
                    break

                recommendation = recommend(prefix.events, path, rules)
                # print(f"{prefix_length} {prefix.trace_num} {prefix.trace_id} {path_index}->{recommendation}")
                trace = input_log[prefix.trace_num]

                if recommendation != "Contradiction" and recommendation != "":
                    # if recommendation != "":
                    selected_path = path
                    trace = input_log[prefix.trace_num]
                    is_compliant, e = evaluate(trace, path, rules, labeling)

                    if e == ConfusionMatrix.TP:
                        eval_res.tp += 1
                    elif e == ConfusionMatrix.FP:
                        eval_res.fp += 1
                    elif e == ConfusionMatrix.FN:
                        eval_res.fn += 1
                    elif e == ConfusionMatrix.TN:
                        eval_res.tn += 1

    try:
        eval_res.precision = eval_res.tp / (eval_res.tp + eval_res.fp)
    except ZeroDivisionError:
        eval_res.precision = 0

    try:
        eval_res.recall = eval_res.tp / (eval_res.tp + eval_res.fn)
    except ZeroDivisionError:
        eval_res.recall = 0

    try:
        eval_res.fscore = 2 * eval_res.precision * eval_res.recall / (eval_res.precision + eval_res.recall)
    except ZeroDivisionError:
        eval_res.fscore = 0

    return eval_res


def generate_recommendations_and_evaluation(test_log, train_log, labeling, prefixing, support_threshold, checkers,
                                            rules, paths):
    if labeling["threshold_type"] == LabelThresholdType.LABEL_MEAN:
        labeling["custom_threshold"] = calc_mean_label_threshold(train_log, labeling)

    target_label = labeling["target"]

    """ Old code without parameters optimization
    (frequent_events, frequent_pairs) = generate_frequent_events_and_pairs(train_log, support_threshold)
    
    print("Generating decision tree input ...")
    dt_input = encode_traces(train_log, frequent_events=frequent_events, frequent_pairs=frequent_pairs, checkers=checkers, rules=rules, labeling=labeling)

    print("Generating decision tree ...")
    paths = generate_decision_tree_paths(dt_input=dt_input, target_label=target_label)
    """

    print("Generating test prefixes ...")
    test_prefixes = generate_prefixes(test_log, prefixing)

    print("Generating recommendations ...")
    recommendations = []
    eval_res = EvaluationResult()
    y = []
    pred = []

    for prefix_length in test_prefixes:
        for prefix in test_prefixes[prefix_length]:
            
            for path in paths:
                path.fitness = calcPathFitnessOnPrefix(prefix.events, path, rules)

            paths = sorted(paths, key=lambda path: (- path.fitness, path.impurity, - path.num_samples["total"]), reverse=False)

            selected_path = None

            for path_index, path in enumerate(paths):

                if selected_path and (path.fitness != selected_path.fitness or path.impurity != selected_path.impurity or path.num_samples != selected_path.num_samples):
                    break

                recommendation = recommend(prefix.events, path, rules)

                #print(f"{prefix_length} {prefix.trace_num} {prefix.trace_id} {path_index}->{recommendation}")
                
                trace = test_log[prefix.trace_num]

                if recommendation != "Contradiction" and recommendation != "":
                    # if recommendation != "":
                    selected_path = path
                    trace = test_log[prefix.trace_num]
                    is_compliant, e = evaluate(trace, path, rules, labeling)

                    """
                    if prefix_length > 5:
                        pdb.set_trace()
                        # for event in prefix.events: print(event['concept:name'])
                        # for path in paths: print(path.fitness)
                        #recommend(prefix.events, paths[0], rules)
                        #len(paths[0])
                        #for rule in paths[0]: print(rule)
                        for path in paths: print(evaluate(trace, path, rules, labeling))
                    """


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
                        target_label=target_label.name,
                        is_compliant=str(is_compliant).upper(),
                        confusion_matrix=e.name,
                        impurity=path.impurity,
                        num_samples=path.num_samples,
                        recommendation=recommendation
                    )
                    y.append(recommendation_model.actual_label)
                    pred.append(
                       recommendation_model.num_samples["positive"] / recommendation_model.num_samples["total"])
                    recommendations.append(recommendation_model)

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

    try:
        fpr, tpr, thresholds = metrics.roc_curve(np.array(y), np.array(pred), pos_label=target_label.name)
        eval_res.auc = metrics.auc(fpr, tpr)
    except:
        eval_res.auc = 0

    print("Writing evaluation result into csv file ...")
    write_evaluation_to_csv(eval_res)

    print("Writing recommendations into csv file ...")
    write_recommendations_to_csv(recommendations)

    return recommendations, eval_res


def write_evaluation_to_csv(e):
    csv_file = "./media/output/result/evaluation.csv"
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
    csv_file = "./media/output/result/recommendations.csv"
    fieldnames = ["Trace id", "Prefix len", "Complete trace", "Current prefix", "Recommendation", "Actual label",
                  "Target label", "Compliant", "Confusion matrix", "Impurity", "Num samples"]
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
