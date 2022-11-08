import numpy as np
import pandas as pd
import pdb
import graphviz
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
import math
from sklearn.tree import DecisionTreeClassifier
from src.models import *
from src.enums import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
import settings
from sklearn.model_selection import GridSearchCV
from src.machine_learning.apriori import *
from src.machine_learning.encoding import *
from src.machine_learning.utils import *
from pm4py.objects.log import obj as log
from pm4py.objects.log.util.get_prefixes import get_log_with_log_prefixes


def find_best_dt(dataset_name, constr_family, data, checkers, rules, labeling, support_threshold_dict, render_dt, num_feat_strategy):
    print("DT params optimization ...")
    categories = [TraceLabel.FALSE.value, TraceLabel.TRUE.value]
    model_dict = {'dataset_name': dataset_name, 'constr_family': constr_family, 'parameters': (),
                  'f1_score_val': None, 'f1_score_train': None, 'f1_prefix_val': None, 'max_depth': 0,
                  'model': None}

    (frequent_events_train, frequent_pairs_train) = generate_frequent_events_and_pairs(data,
                                                                                       support_threshold_dict['min'])

    if support_threshold_dict['max'] <= 1.0:
        max_frequent_events_train = get_frequent_events(data, support_threshold_dict['max'])
        set_freq_events_max = set(max_frequent_events_train)
        freq_event_diff = [x for x in frequent_events_train if x not in set_freq_events_max]

        if len(freq_event_diff) > 0:
            # Making event pairs ...
            pairs = list(combinations(freq_event_diff, 2))
            # Finding frequent pairs
            frequent_pairs = get_frequent_pairs(data, pairs, support_threshold_dict['min'])

            all_frequent_pairs = []
            for pair in frequent_pairs:
                (x, y) = pair
                reverse_pair = (y, x)
                all_frequent_pairs.extend([pair, reverse_pair])

    # Generating decision tree input
    if settings.train_prefix_log:
        """
        data_log = log.EventLog()
        trainval_prefixes = generate_prefixes(data, prefixing)['UPTO']
        for prefix in trainval_prefixes:
            trace_xes = log.Trace()
            #pdb.set_trace()
            trace_xes.attributes["concept:name"] = prefix.trace_id
            for event in prefix.events:
                event_xes = log.Event()
                #pdb.set_trace()
                event_xes["concept:name"] = event["concept:name"]
                event_xes["time:timestamp"] = event["time:timestamp"]
                event_xes["label"] = event["label"]
                trace_xes.append(event_xes)
            data_log.append(trace_xes)
        """
        prefix_log, trace_ids = get_log_with_log_prefixes(data)
        data = log.EventLog()
        for trace in prefix_log:
            if len(trace) > 2:
                data.append(trace)

    dt_input_trainval = encode_traces(data, frequent_events=frequent_events_train, frequent_pairs=frequent_pairs_train,
                                      checkers=checkers, rules=rules, labeling=labeling)

    X_train = pd.DataFrame(dt_input_trainval.encoded_data, columns=dt_input_trainval.features)
    y_train = pd.Categorical(dt_input_trainval.labels, categories=categories)

    if num_feat_strategy == 'sqrt':
        num_feat = int(math.sqrt(len(dt_input_trainval.features)))
    else:
        num_feat = int(num_feat_strategy * len(dt_input_trainval.features))

    sel = SelectKBest(mutual_info_classif, k=num_feat)
    X_train = sel.fit_transform(X_train, y_train)
    cols = sel.get_support(indices=True)
    new_feature_names = np.array(dt_input_trainval.features)[cols]

    print("Grid search ...")
    search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=settings.dt_hyperparameters, scoring="f1", return_train_score=True, cv=5)
    search.fit(X_train, y_train)

    model_dict['model'] = search.best_estimator_
    f1_score_train = round(100*search.cv_results_['mean_train_score'][search.best_index_], 2)
    model_dict['f1_score_val'] = round(100*search.best_score_, 2)
    model_dict['f1_score_train'] = f1_score_train
    model_dict['f1_prefix_val'] = -1
    model_dict['max_depth'] = search.best_estimator_.tree_.max_depth
    model_dict['parameters'] = tuple(search.best_params_.values())

    if render_dt:
        dot_data = tree.export_graphviz(search.best_estimator_, out_file=None, impurity=True,
                                        feature_names=new_feature_names, node_ids=True, filled=True)
                                        # class_names=['regular', 'deviant'])
        graph = graphviz.Source(dot_data, format="pdf")
        graph.render(os.path.join(settings.output_dir, f'DT_{dataset_name}_{constr_family}'))
    return model_dict, new_feature_names


def dt_score(dt_input):
    categories = [TraceLabel.FALSE.value, TraceLabel.TRUE.value]

    X = pd.DataFrame(dt_input.encoded_data, columns=dt_input.features)
    y = pd.Categorical(dt_input.labels, categories=categories)
    dtc = DecisionTreeClassifier(class_weight=None, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    # np.sum(y.to_numpy())
    return f1_score(y_test, y_pred)


def generate_boost_decision_tree(X_train, X_val, y_train, y_val, class_weight, min_samples_split):
    # dtc = DecisionTreeClassifier(min_samples_split=min_samples_split, class_weight=class_weight, random_state=0)
    dtc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0).fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_val)
    # np.sum(y.to_numpy())
    return dtc.estimators_[0, 0], f1_score(y_val, y_pred)


def generate_decision_tree(X_train, X_val, y_train, y_val, class_weight, min_samples_split, use_smote=False):
    count = Counter(y_train)
    pos_ratio = count[TraceLabel.TRUE.value] / count[TraceLabel.FALSE.value]

    if use_smote and pos_ratio <= 0.4:
        sm = SMOTE()
        sme = SMOTEENN()
        ada = ADASYN()
        X_train, y_train = sm.fit_resample(X_train, y_train)

    dtc = DecisionTreeClassifier(min_samples_split=min_samples_split, class_weight=class_weight, random_state=0)
    # dtc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
    # dtc = RandomForestClassifier(class_weight=class_weight)
    # dtc = SVC(class_weight=class_weight)
    # dtc = MLPClassifier(random_state=1, max_iter=1000)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_val)
    y_pred_train = dtc.predict(X_train)
    # np.sum(y.to_numpy())
    # pdb.set_trace()
    return dtc, f1_score(y_val, y_pred), f1_score(y_train, y_pred_train)


def generate_paths(dtc, dt_input_features, target_label):
    # Finding decision tree paths
    left = dtc.tree_.children_left
    right = dtc.tree_.children_right
    features = [dt_input_features[i] for i in dtc.tree_.feature]
    leaf_ids = np.argwhere(left == -1)[:, 0]
    if target_label == TraceLabel.TRUE:
        leaf_ids_positive = filter(
            lambda leaf_id: dtc.tree_.value[leaf_id][0][0] < dtc.tree_.value[leaf_id][0][1], leaf_ids)
    else:
        leaf_ids_positive = filter(
            lambda leaf_id: dtc.tree_.value[leaf_id][0][0] > dtc.tree_.value[leaf_id][0][1], leaf_ids)

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            state = TraceState.VIOLATED
        else:
            parent = np.where(right == child)[0].item()
            state = TraceState.SATISFIED

        lineage.append((features[parent], state, parent))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    paths = []
    for leaf_id in leaf_ids_positive:
        rules = []
        for node in recurse(left, right, leaf_id):
            rules.append(node)
        if target_label == TraceLabel.TRUE:
            num_samples = {
                "node_samples": dtc.tree_.n_node_samples[leaf_id],
                "negative": dtc.tree_.value[leaf_id][0][0],
                "positive": dtc.tree_.value[leaf_id][0][1],
                "total": dtc.tree_.value[leaf_id][0][0] + dtc.tree_.value[leaf_id][0][1]
            }
        else:
            num_samples = {
                "node_samples": dtc.tree_.n_node_samples[leaf_id],
                "negative": dtc.tree_.value[leaf_id][0][1],
                "positive": dtc.tree_.value[leaf_id][0][0],
                "total": dtc.tree_.value[leaf_id][0][0] + dtc.tree_.value[leaf_id][0][1]
            }
        path = PathModel(
            impurity=dtc.tree_.impurity[leaf_id],
            num_samples=num_samples,
            rules=rules
        )
        paths.append(path)
    return paths


def generate_decision_tree_paths(dt_input, target_label):
    categories = [TraceLabel.FALSE.value, TraceLabel.TRUE.value]

    X = pd.DataFrame(dt_input.encoded_data, columns=dt_input.features)
    y = pd.Categorical(dt_input.labels, categories=categories)
    dtc = DecisionTreeClassifier(class_weight=None, random_state=0)
    dtc.fit(X, y)

    # find paths
    print("Finding decision tree paths ...")
    left = dtc.tree_.children_left
    right = dtc.tree_.children_right
    features = [dt_input.features[i] for i in dtc.tree_.feature]
    leaf_ids = np.argwhere(left == -1)[:, 0]
    if target_label == TraceLabel.TRUE:
        leaf_ids_positive = filter(
            lambda leaf_id: dtc.tree_.value[leaf_id][0][0] < dtc.tree_.value[leaf_id][0][1], leaf_ids)
    else:
        leaf_ids_positive = filter(
            lambda leaf_id: dtc.tree_.value[leaf_id][0][0] > dtc.tree_.value[leaf_id][0][1], leaf_ids)

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            state = TraceState.VIOLATED
        else:
            parent = np.where(right == child)[0].item()
            state = TraceState.SATISFIED

        lineage.append((features[parent], state))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    paths = []
    for leaf_id in leaf_ids_positive:
        rules = []
        for node in recurse(left, right, leaf_id):
            rules.append(node)
        if target_label == TraceLabel.TRUE:
            num_samples = {
                "negative": dtc.tree_.value[leaf_id][0][0],
                "positive": dtc.tree_.value[leaf_id][0][1],
                "total": dtc.tree_.value[leaf_id][0][0] + dtc.tree_.value[leaf_id][0][1]
            }
        else:
            num_samples = {
                "negative": dtc.tree_.value[leaf_id][0][1],
                "positive": dtc.tree_.value[leaf_id][0][0],
                "total": dtc.tree_.value[leaf_id][0][0] + dtc.tree_.value[leaf_id][0][1]
            }
        path = PathModel(
            impurity=dtc.tree_.impurity[leaf_id],
            num_samples=num_samples,
            rules=rules
        )
        paths.append(path)
    return paths
