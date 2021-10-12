import numpy as np
import pandas as pd
import pdb
from sklearn.tree import DecisionTreeClassifier
from src.models import *
from src.enums import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


def dt_score(dt_input):
    categories = [TraceLabel.FALSE.value, TraceLabel.TRUE.value]

    X = pd.DataFrame(dt_input.encoded_data, columns=dt_input.features)
    y = pd.Categorical(dt_input.labels, categories=categories)
    dtc = DecisionTreeClassifier(class_weight=None, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    # np.sum(y.to_numpy())
    return f1_score(y_test, y_pred)

def generate_boost_decision_tree(X_train, X_val, y_train, y_val, class_weight, min_samples_split):
    #dtc = DecisionTreeClassifier(min_samples_split=min_samples_split, class_weight=class_weight, random_state=0)
    dtc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0).fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_val)
    # np.sum(y.to_numpy())
    return dtc.estimators_[0, 0], f1_score(y_val, y_pred)


def generate_decision_tree(X_train, X_val, y_train, y_val, class_weight, min_samples_split):
    dtc = DecisionTreeClassifier(min_samples_split=min_samples_split, class_weight=class_weight, random_state=0)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_val)
    y_pred_train = dtc.predict(X_train)
    # np.sum(y.to_numpy())
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
