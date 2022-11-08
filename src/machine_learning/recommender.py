import copy
#from src.machine_learning.utils import *
#from src.machine_learning.apriori import *
#from src.machine_learning.encoding import *
from src.machine_learning.decision_tree import *
from src.models import EvaluationResult
from src.constants import *
import csv
import numpy as np
import settings
from sklearn import metrics


class ParamsOptimizer:
    def __init__(self, data_log, train_val_log, val_log, train_log, parameters, labeling, checkers, rules, min_prefix_length, max_prefix_length):
        self.parameter_names = parameters.keys()
        self.val_log = val_log
        self.data_log = data_log
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
                          'id': param_id, 'model': None, 'frequent_events': None, 'frequent_pairs': None}

            (frequent_events_train, frequent_pairs_train) = generate_frequent_events_and_pairs(self.data_log,
                                                                                               param_tuple[0])

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
            model_dict['f1_score_val'] = f1_score_val
            model_dict['f1_score_train'] = f1_score_train
            model_dict['f1_prefix_val'] = np.average([res.fscore for res in results])
            model_dict['frequent_events'] = frequent_events_train
            model_dict['frequent_pairs'] = frequent_pairs_train
            self.model_grid.append(model_dict)

        # retrain the DT using train_val set with the best params tested on the val set
        sorted_models = sorted(self.model_grid, key=lambda d: d['f1_prefix_val'])
        best_model_dict = sorted_models[-1]

        #
        # best_model_dict['frequent_pairs']

        # (frequent_events_trainval, frequent_pairs_trainval) = generate_frequent_events_and_pairs(self.train_val_log,
        #                                                                                   best_model_dict['parameters'][0])
        dt_input_trainval = encode_traces(self.train_val_log, frequent_events=best_model_dict['frequent_events'],
                                          frequent_pairs=best_model_dict['frequent_pairs'], checkers=self.checkers,
                                          rules=self.rules, labeling=self.labeling)
        dt_input_val = encode_traces(self.val_log, frequent_events=best_model_dict['frequent_events'],
                                     frequent_pairs=best_model_dict['frequent_pairs'], checkers=self.checkers,
                                     rules=self.rules, labeling=self.labeling)

        X_train_val = pd.DataFrame(dt_input_trainval.encoded_data, columns=dt_input_trainval.features)
        y_train_val = pd.Categorical(dt_input_trainval.labels, categories=categories)
        X_val = pd.DataFrame(dt_input_val.encoded_data, columns=dt_input_val.features)
        y_val = pd.Categorical(dt_input_val.labels, categories=categories)

        dtc, _, _ = generate_decision_tree(X_train_val, X_val, y_train_val, y_val,
                                           class_weight=best_model_dict['parameters'][1],
                                           min_samples_split=best_model_dict['parameters'][2])
        best_model_dict['model'] = dtc
        best_model_dict['max_depth'] = dtc.tree_.max_depth

        del best_model_dict["frequent_events"]
        del best_model_dict["frequent_pairs"]
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
            # X_new = SelectKBest(mutual_info_classif, k=int(0.7*X.shape[1])).fit_transform(X, y)

            # Tree
            # clf = ExtraTreesClassifier(n_estimators=50)
            # clf = clf.fit(X, y)
            # model = SelectFromModel(clf, prefit=True)
            # X_new = model.transform(X)

            X_new = X
            # Create the RFE object and compute a cross-validated score.
            # svc = SVC(kernel="linear")
            # rfe = RFE(estimator=svc, step=1)
            # rfe.fit(X, y)
            # X_new = rfe.transform(X)

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
        template, rule_state, _ = rule
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


def evaluate_OLD(trace, path, rules, labeling):
    # Compliantness 0/1
    is_compliant = True
    for rule in path.rules:
        template, rule_state, _, _ = rule
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
            # print(f"1,{label},{cm}")
        else:
            cm = ConfusionMatrix.FN if label == TraceLabel.TRUE else ConfusionMatrix.TN
            # print(f"0,{label},{cm}")
    else:
        print("---------------------")
        if is_compliant:
            cm = ConfusionMatrix.FN if label == TraceLabel.TRUE else ConfusionMatrix.TN
        else:
            cm = ConfusionMatrix.TP if label == TraceLabel.TRUE else ConfusionMatrix.FP
    return is_compliant, cm


def evaluate(trace, path, rules, labeling, sat_threshold, eval_type='strong'):
    # Compliantness con different strategies
    is_compliant = True
    rule_occurencies = 0
    rule_activations = []
    for rule in path.rules:
        template, rule_state, _ = rule
        template_name, template_params = parse_method(template)

        result = None
        if template_name in [ConstraintChecker.EXISTENCE.value, ConstraintChecker.ABSENCE.value, ConstraintChecker.INIT.value, ConstraintChecker.EXACTLY.value]:
            result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](trace, True, template_params[0], rules)
        else:
            result = CONSTRAINT_CHECKER_FUNCTIONS[template_name](trace, True, template_params[0], template_params[1], rules)

        if eval_type == 'count_activations':
            # Existence templates
            if result.num_fulfillments is None:
                if rule_state == result.state:
                    rule_activations.append(1)
                else:
                    rule_activations.append(0)
            # Other templates
            else:
                if result.num_activations > 0:
                    rule_activations.append(result.num_fulfillments/result.num_activations)
                else:
                    rule_activations.append(1)

        elif eval_type == 'count_occurrences':
            if rule_state == result.state:
                rule_occurencies += 1
        else:
            if rule_state != result.state:
                is_compliant = False
                break

    if eval_type == 'count_activations':
        is_compliant = True if np.mean(rule_activations) > sat_threshold else False
    elif eval_type == 'count_occurrences':
        is_compliant = True if rule_occurencies / len(path.rules) > sat_threshold else False

    label = generate_label(trace, labeling)

    if labeling["target"] == TraceLabel.TRUE:
        if is_compliant:
            cm = ConfusionMatrix.TP if label == TraceLabel.TRUE else ConfusionMatrix.FP
            # print(f"1,{label},{cm}")
        else:
            cm = ConfusionMatrix.FN if label == TraceLabel.TRUE else ConfusionMatrix.TN
            # print(f"0,{label},{cm}")
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


def train_path_recommender(data_log, train_val_log, val_log, train_log, labeling, support_threshold, checkers, rules,
                           dataset_name, constr_family, output_dir, min_prefix_length, max_prefix_length, feat_strategy):
    if labeling["threshold_type"] == LabelThresholdType.LABEL_MEAN:
        labeling["custom_threshold"] = calc_mean_label_threshold(train_log, labeling)

    target_label = labeling["target"]

    if dataset_name == 'traffic_fines_1':
        settings.hyperparameters['class_weight'] = [None]
        settings.dt_hyperparameters['class_weight'] = [None]

    print("Generating decision tree with params optimization ...")
    if settings.optmize_dt:
        best_model_dict, feature_names = find_best_dt(dataset_name, constr_family, train_val_log, checkers, rules,
                                                      labeling, support_threshold, settings.print_dt, feat_strategy)
    else:
        param_opt = ParamsOptimizer(data_log, train_val_log, val_log, train_log, settings.hyperparameters, labeling,
                                    checkers, rules, min_prefix_length, max_prefix_length)
        best_model_dict, feature_names = param_opt.params_grid_search(dataset_name, constr_family)

    with open(os.path.join(output_dir, 'model_params.csv'), 'a') as f:
        w = csv.writer(f, delimiter='\t')
        row = list(best_model_dict.values())
        w.writerow(row[:-1]) # do not print the model

    print("Generating decision tree paths ...")
    paths = generate_paths(dtc=best_model_dict['model'], dt_input_features=feature_names,
                           target_label=target_label)
    return paths


def evaluate_recommendations(input_log, labeling, prefixing, rules, paths):
    # if labeling["threshold_type"] == LabelThresholdType.LABEL_MEAN:
    #    labeling["custom_threshold"] = calc_mean_label_threshold(train_log, labeling)

    target_label = labeling["target"]

    print("Generating test prefixes ...")
    prefixes = generate_prefixes(input_log, prefixing)

    eval_res = EvaluationResult()

    for prefix_length in prefixes:
        # for id, pref in enumerate(prefixes[prefix_length]): print(id, input_log[pref.trace_num][0]['label'])
        for prefix in prefixes[prefix_length]:
            for path in paths:
                path.fitness = calcPathFitnessOnPrefix(prefix.events, path, rules, settings.fitness_type)

            paths = sorted(paths, key=lambda path: (- path.fitness, path.impurity, - path.num_samples["total"]), reverse=False)

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

                    """
                    if prefix_length > 2:
                        # for event in prefix.events: print(event['concept:name'])
                        # for path in paths: print(path.rules)
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
                                            rules, paths, dataset_name, hyperparams_evaluation, eval_res=None, debug=False):
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
    if eval_res is None:
        eval_res = EvaluationResult()
    y = []
    pred = []

    for prefix_length in test_prefixes:
        eval_res.prefix_length = prefix_length
        # for id, pref in enumerate(test_prefixes[prefix_length]): print(id, test_log[pref.trace_num][0]['label'])
        for prefix in test_prefixes[prefix_length]:
            eval_res.num_cases = len(test_prefixes[prefix_length])

            pos_paths_total_samples = 0
            for path in paths:
                pos_paths_total_samples += path.num_samples['node_samples']
            for path in paths:
                path.fitness = calcPathFitnessOnPrefix(prefix.events, path, rules, settings.fitness_type)
                path.score = calcScore(path, pos_paths_total_samples, weights=hyperparams_evaluation[1:])

            # paths = sorted(paths, key=lambda path: (- path.fitness, path.impurity, - path.num_samples["total"]), reverse=False)
            if settings.use_score:
                paths = sorted(paths, key=lambda path: (- path.score), reverse=False)
            else:
                paths = sorted(paths, key=lambda path: (- path.fitness), reverse=False)

            reranked_paths = copy.deepcopy(paths)
            if settings.reranking:
                reranked_paths = paths[:settings.top_K_paths]
                reranked_paths = sorted(reranked_paths, key=lambda path: (- path.score), reverse=False)

            if settings.compute_gain and len(reranked_paths) > 0:
                raw_prefix = [event['concept:name'] for event in prefix.events]
                trace = test_log[prefix.trace_num]
                path = reranked_paths[0]
                label = generate_label(trace, labeling)
                compliant, _ = evaluate(trace, path, rules, labeling, eval_type=settings.sat_type)
                eval_res.comp += 1 if compliant else 0
                eval_res.non_comp += 0 if compliant else 1
                eval_res.pos_comp += 1 if compliant and label.value == TraceLabel.TRUE.value else 0
                eval_res.pos_non_comp += 1 if not compliant and label.value == TraceLabel.TRUE.value else 0

            selected_path = None
            for path_index, path in enumerate(reranked_paths):

                if selected_path and (path.fitness != selected_path.fitness or path.impurity != selected_path.impurity
                                      or path.num_samples != selected_path.num_samples):
                    break

                recommendation = recommend(prefix.events, path, rules)
                # print(f"{prefix_length} {prefix.trace_num} {prefix.trace_id} {path_index}->{recommendation}")

                if recommendation != "Contradiction" and recommendation != "":
                    # if True:
                    # if recommendation != "":

                    selected_path = path
                    trace = test_log[prefix.trace_num]
                    #print(prefix.trace_id, trace[0]['label'])
                    is_compliant, e = evaluate(trace, path, rules, labeling, sat_threshold=hyperparams_evaluation[0],
                                               eval_type=settings.sat_type)
                    #if prefix_length == 12 or prefix_length == 12:
                        #pdb.set_trace()
                    #pdb.set_trace()
                    if debug:
                        #pdb.set_trace()
                        if prefix.trace_id == 'GX' and prefix_length == 15:
                            for event in prefix.events: print(event['concept:name'])
                            #pdb.set_trace()
                        if prefix.trace_id == 'DS' and prefix_length == 5:
                            for event in prefix.events: print(event['concept:name'])
                            print(e)
                            print(prefix_length)
                            #pdb.set_trace()
                        """
                        if len(recommendation) > 50:
                            print(e)
                            print(prefix.trace_id)
                            print(prefix_length)
                            pdb.set_trace()
                        """

                    """
                    if prefix_length > 2:
                        is_compliant, e = evaluate(trace, path, rules, labeling)
                        # c
                        # for path in paths: print(f"{path.fitness:.2f}, {1 - path.impurity:.2f}, {path.num_samples['node_samples']:.2f}")
                        # for path in reranked_paths: print(f"{path.fitness:.2f}, {1 - path.impurity:.2f}, {path.num_samples['positive']/path.num_samples['total']:.2f}")
                        #recommend(test_prefixes[prefix_length][idd].events, paths[0], rules)
                        #len(paths[0])
                        #for rule in paths[0].rules: print(rule)
                        # paths[0].rules
                        len(test_prefixes[prefix_length])
                        for path in paths: print(evaluate(trace, path, rules, labeling))
                        for idx, path in enumerate(reranked_paths): print(idx, evaluate(trace, path, rules, labeling))
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
                        fitness=path.fitness,
                        score=path.score,
                        recommendation=recommendation
                    )
                    y.append(recommendation_model.actual_label)
                    pred.append(recommendation_model.num_samples["positive"]/recommendation_model.num_samples["total"])
                    # pred.append(recommendation_model.score)
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

    try:
        eval_res.mcc = matthews_corrcoef(eval_res.tp, eval_res.fp, eval_res.fn, eval_res.tn)
    except:
        eval_res.mcc = 0

    if settings.compute_gain:
        eval_res.gain = gain(eval_res.comp, eval_res.non_comp, eval_res.pos_comp, eval_res.pos_non_comp)

    # print("Writing evaluation result into csv file ...")
    # write_evaluation_to_csv(eval_res, dataset_name)

    # print("Writing recommendations into csv file ...")
    # write_recommendations_to_csv(recommendations, dataset_name)

    return recommendations, eval_res


def write_evaluation_to_csv(e, dataset):
    csv_file = os.path.join(settings.results_dir, f"{dataset}_evaluation.csv")
    fieldnames = ["tp", "fp", "tn", "fn", "precision", "recall", "accuracy", "fscore", "auc"]
    values = {
        "tp": e.tp,
        "fp": e.fp,
        "tn": e.tn,
        "fn": e.fn,
        "precision": round(e.precision, 2),
        "recall": round(e.recall, 2),
        "accuracy": round(e.accuracy, 2),
        "fscore": round(e.fscore, 2),
        "auc": round(e.auc, 2)
    }
    try:
        with open(csv_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(values)
    except IOError:
        print("I/O error")


def write_recommendations_to_csv(recommendations, dataset):
    csv_file = os.path.join(settings.results_dir, f"{dataset}_recommendations.csv")
    fieldnames = ["Trace id", "Prefix len", "Complete trace", "Current prefix", "Recommendation", "Actual label",
                  "Target label", "Compliant", "Confusion matrix", "Impurity", "Fitness", "Num samples"]
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
                "Fitness": r.fitness,
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


def prefix_evaluation_to_csv(result_dict, dataset):
    csv_file = os.path.join(settings.results_dir, f"{dataset}_evaluation.csv")
    fieldnames = ["prefix_length", "num_cases"]
    basic_fields = ["comp", "non_comp", "pos_comp", "pos_non_comp", "tp", "fp", "tn", "fn", "precision", "recall", "fscore"]
    for constr_family in result_dict.keys():
        fieldnames += [f"{constr_family}_{field}" for field in basic_fields]

    try:
        with open(csv_file, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(fieldnames)
            res_dict = {}
            for constr_family in result_dict:
                res_dict[constr_family] = []
                for eval_obj in result_dict[constr_family]:
                    res_dict[constr_family].append([eval_obj.prefix_length, eval_obj.num_cases] +
                                                    [getattr(eval_obj, field) for field in basic_fields])

            table_res = res_dict[list(res_dict.keys())[0]]
            for constr_family in list(res_dict.keys())[1:]:
                table_res = np.hstack((table_res, np.array(res_dict[constr_family])[:, 2:]))

            for row in table_res:
                writer.writerow(row)

    except IOError:
        print("I/O error")