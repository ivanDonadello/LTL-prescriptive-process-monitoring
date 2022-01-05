import os
import pdb
from src.enums.ConstraintChecker import ConstraintChecker

# ================ thresholds ================
support_threshold_dict = {'min': 0.15, 'max': 1.75}
sat_threshold = 0.75
top_K_paths = 6
reranking = False
sat_type = 'count_occurrences'  # count_occurrences or count_activations or strong
fitness_type = 'mean'  # mean or wmean
cumulative_res = True
optmize_dt = True
print_dt = True
compute_gain = True
smooth_factor = 1
num_classes = 2
train_prefix_log = False
one_hot_encoding = True
use_score = False

# ================ folders ================
output_dir = "media/output"
results_dir = os.path.join(output_dir, "result")
dataset_folder = "media/input/processed_benchmark_event_logs"

# ================ checkers ================
existence_family = [ConstraintChecker.EXISTENCE, ConstraintChecker.ABSENCE, ConstraintChecker.INIT,
                    ConstraintChecker.EXACTLY]

choice_family = [ConstraintChecker.CHOICE, ConstraintChecker.EXCLUSIVE_CHOICE]

positive_rel_family = [ConstraintChecker.RESPONDED_EXISTENCE, ConstraintChecker.RESPONSE,
                       ConstraintChecker.ALTERNATE_RESPONSE, ConstraintChecker.CHAIN_RESPONSE,
                       ConstraintChecker.PRECEDENCE, ConstraintChecker.ALTERNATE_PRECEDENCE,
                       ConstraintChecker.CHAIN_PRECEDENCE]

negative_rel_family = [ConstraintChecker.NOT_RESPONDED_EXISTENCE, ConstraintChecker.NOT_RESPONSE,
                       ConstraintChecker.NOT_CHAIN_RESPONSE, ConstraintChecker.NOT_PRECEDENCE,
                       ConstraintChecker.NOT_CHAIN_PRECEDENCE]

checkers_cumulative = {"existence": existence_family}
checkers_cumulative["choice"] = checkers_cumulative["existence"] + choice_family
checkers_cumulative["positive relations"] = checkers_cumulative["choice"] + positive_rel_family
checkers_cumulative["negative relations"] = checkers_cumulative["positive relations"] + negative_rel_family

checkers = {"existence": existence_family,
            "choice": existence_family + choice_family,
            "positive relations": existence_family + positive_rel_family,
            "negative relations": existence_family + negative_rel_family,
            "all": checkers_cumulative['negative relations']}

constr_family_list = ["existence", "choice"]  # "choice", "positive relations", "negative relations"]
constr_family_list = checkers.keys()

# ================ datasets ================
datasets_names = ["bpic2011_f1", "bpic2011_f2", "bpic2011_f3", "bpic2011_f4",
                  "bpic2015_1_f2", "bpic2015_2_f2", "bpic2015_3_f2", "bpic2015_4_f2",
                  "bpic2015_5_f2", "bpic2017_accepted", "bpic2017_cancelled",
                  "bpic2017_refused", "bpic2012_cancelled",
                  "bpic2012_accepted", "bpic2012_declined",
                  "hospital_billing_2", "hospital_billing_3", "Production",
                  "sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4", "traffic_fines_1"]
# datasets_names = ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4", "Production"]
# datasets_names = ["bpic2015_1_f2"]

# ================ hyperparameters ================
hyperparameters = {'support_threshold': [support_threshold_dict['min']-0.2, support_threshold_dict['min']-0.1,
                                         support_threshold_dict['min'],
                                         support_threshold_dict['min']+0.1],
                   'class_weight': [None, 'balanced'],
                   'min_samples_split': [2]}

dt_hyperparameters = {'criterion': ['gini', 'entropy'],
                      'class_weight': ['balanced', None],
                      'max_depth': [4, 6, 8, 10],
                      'min_samples_split': [2, 0.1, 0.2, 0.3],
                      'min_samples_leaf': [1, 10, 16]}

dt_hyperparameters = {'criterion': ['entropy', 'gini'],
                      'class_weight': ['balanced', None],
                      'max_depth': [4, 6, 8, 10, None],
                      'min_samples_split': [0.1, 2, 0.2, 0.3],
                      'min_samples_leaf': [10, 1, 16]}
"""
hyperparameters = {'support_threshold': [support_threshold],
                 'class_weight': [None],
                  'min_samples_split': [2]}
"""

# ================ checkers satisfaction ================
rules = {
    "vacuous_satisfaction": True,
    "activation": "",  # e.g. A.attr > 6
    "correlation": "",  # e.g. T.attr < 12
    "n": {
        ConstraintChecker.EXISTENCE: 1,
        ConstraintChecker.ABSENCE: 1,
        ConstraintChecker.EXACTLY: 1,
    }
}

# ================ plots ================
method_label = {'existence': r'$\mathcal{E}$', 'choice': r'$\mathcal{C}$', 'positive relations': r'$\mathcal{PR}$',
                'negative relations': r'$\mathcal{NR}$', 'all': r'$\mathcal{A}$'}
method_marker = {'existence': '<', 'choice': '1', 'positive relations': '.', 'negative relations': '', 'all': '+'}
method_color = {'existence': 'mediumpurple', 'choice': 'deepskyblue', 'positive relations': 'orange',
                'negative relations': 'crimson', 'all': 'forestgreen'}