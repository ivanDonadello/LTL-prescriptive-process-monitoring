import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from IPython.display import Image

from declare_based.src.models import *
from declare_based.src.enums import *


def generate_decision_tree_paths(dt_input, target_label):
    categories = [TraceLabel.FALSE.value, TraceLabel.TRUE.value]
    res = {}
    for key in dt_input:
        if dt_input[key].prefix_length > 0:
            feature_names = dt_input[key].features
            encoded_data = dt_input[key].encoded_data
            labels = dt_input[key].labels

            X = pd.DataFrame(encoded_data, columns=feature_names)
            y = pd.Categorical(labels, categories=categories)
            dtc = DecisionTreeClassifier(random_state=0)
            dtc.fit(X, y)

            dot_data = tree.export_graphviz(dtc, out_file=None, feature_names=feature_names, class_names=[TraceLabel.FALSE.name, TraceLabel.TRUE.name],
                                            filled=True, rounded=True, special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data)
            Image(graph.create_png())

            # find paths
            left = dtc.tree_.children_left
            right = dtc.tree_.children_right
            features = [feature_names[i] for i in dtc.tree_.feature]
            leaf_ids = np.argwhere(left == -1)[:, 0]
            if len(leaf_ids) == 1 and leaf_ids[0] == 0:
                res[key] = []
            else:
                if target_label == TraceLabel.TRUE.name:
                    leaf_ids_positive = filter(lambda leaf_id: dtc.tree_.value[leaf_id][0][0] < dtc.tree_.value[leaf_id][0][1], leaf_ids)
                else:
                    leaf_ids_positive = filter(lambda leaf_id: dtc.tree_.value[leaf_id][0][0] > dtc.tree_.value[leaf_id][0][1], leaf_ids)

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
                    path = PathModel(impurity=dtc.tree_.impurity[leaf_id],
                                     num_samples=dtc.tree_.value[leaf_id][0][0] + dtc.tree_.value[leaf_id][0][1], rules=rules)
                    paths.append(path)
                res[key] = sorted(paths, key=lambda path: (path.impurity, - path.num_samples), reverse=False)
    return res