import pdb
import sys
import pickle
from src.dataset_manager.datasetManager import DatasetManager
from src.machine_learning import *
from pm4py.objects.conversion.log import converter as log_converter
import argparse
import time
import numpy as np


def rec_sys_exp(dataset_name):
    # generate rules
    settings.rules["activation"] = generate_rules(settings.rules["activation"])
    settings.rules["correlation"] = generate_rules(settings.rules["correlation"])

    dataset_manager = DatasetManager(dataset_name.lower())
    data = dataset_manager.read_dataset(os.path.join(os.getcwd(), settings.dataset_folder))

    # split into training and test
    train_val_ratio = 0.8
    if dataset_name == "bpic2015_4_f2":
        train_val_ratio = 0.85
    train_ratio = 0.8
    train_val_df, test_df = dataset_manager.split_data_strict(data, train_val_ratio)
    train_df, val_df = dataset_manager.split_data(train_val_df, train_ratio, split="random")

    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length_test, max_prefix_length_val = 9, 9
    elif "bpic2017" in dataset_name:
        max_prefix_length_test = min(20, dataset_manager.get_pos_case_length_quantile(test_df, 0.90))
        max_prefix_length_val = min(20, dataset_manager.get_pos_case_length_quantile(val_df, 0.90))
    else:
        max_prefix_length_test = min(40, dataset_manager.get_pos_case_length_quantile(test_df, 0.90))
        max_prefix_length_val = min(40, dataset_manager.get_pos_case_length_quantile(val_df, 0.90))

    data = data.rename(columns={dataset_manager.timestamp_col: 'time:timestamp',
                                dataset_manager.case_id_col: 'case:concept:name',
                                dataset_manager.activity_col: 'concept:name'})

    train_df = train_df.rename(
        columns={dataset_manager.timestamp_col: 'time:timestamp', dataset_manager.case_id_col: 'case:concept:name',
                 dataset_manager.activity_col: 'concept:name'})
    test_df = test_df.rename(
        columns={dataset_manager.timestamp_col: 'time:timestamp', dataset_manager.case_id_col: 'case:concept:name',
                 dataset_manager.activity_col: 'concept:name'})
    val_df = val_df.rename(
        columns={dataset_manager.timestamp_col: 'time:timestamp', dataset_manager.case_id_col: 'case:concept:name',
                 dataset_manager.activity_col: 'concept:name'})
    train_val_df = train_val_df.rename(
        columns={dataset_manager.timestamp_col: 'time:timestamp', dataset_manager.case_id_col: 'case:concept:name',
                 dataset_manager.activity_col: 'concept:name'})
    val_log = log_converter.apply(val_df)
    train_log = log_converter.apply(train_df)
    test_log = log_converter.apply(test_df)
    train_val_log = log_converter.apply(train_val_df)
    data_log = log_converter.apply(data)

    labeling = {
        "type": LabelType.TRACE_CATEGORICAL_ATTRIBUTES,
        "threshold_type": "",
        "target": TraceLabel.TRUE,  # lower than a threshold considered as True
        "trace_lbl_attr": dataset_manager.label_col,
        "trace_label": dataset_manager.pos_label,
        "custom_threshold": 0.0
    }
    """
    labeling = {
        "type": LabelType.TRACE_CATEGORICAL_ATTRIBUTES,
        "threshold_type": "",
        "target": TraceLabel.TRUE,  # lower than a threshold considered as True
        "trace_lbl_attr": dataset_manager.label_col,
        "trace_label": 'regular',
        "custom_threshold": 0.0
    }
    
    labeling = {
        "type": LabelType.TRACE_DURATION,
        "threshold_type": LabelThresholdType.LABEL_MEAN,
        "target": TraceLabel.TRUE,  # lower than a threshold considered as True
        "trace_attribute": "",
        "custom_threshold": 0.0
    }
    """

    # generate recommendations and evaluation
    results = {family: [] for family in settings.constr_family_list}
    time_results = [["dataset_name", "constr_family", "case_id", "prefix_length", "total_recommendation_time"]]
    train_opt = [["dataset_name", "constr_family", "train_opt_time"]]

    for constr_family in settings.constr_family_list:
        start_train_opt_time = timeit.default_timer()
        prefix_lenght_list_test = list(range(min_prefix_length, max_prefix_length_test + 1))
        prefix_lenght_list_val = list(range(min_prefix_length, max_prefix_length_val + 1))

        if load_model:
            try:
                with open(os.path.join(settings.models_path, f'{dataset_name}_{constr_family}.pickle'), 'rb') as file:
                    paths, best_hyperparams_combination = pickle.load(file)
                    print(f"Model {dataset_name}_{constr_family}.pickle loaded")
            except FileNotFoundError as not_found:
                print(f"Model {dataset_name}_{constr_family}.pickle not found. Invalid path or you "
                      f"have to train a model before loading.")
                sys.exit(2)
        else:
            feat_strategy_paths_dict = {strategy: None for strategy in settings.num_feat_strategy}
            hyperparams_evaluation_list = []
            results_hyperparams_evaluation = {}
            hyperparams_evaluation_list_baseline = []

            for v1 in settings.sat_threshold_list:
                # the baseline chooses the path with highest probability
                hyperparams_evaluation_list_baseline.append((v1,) + (0, 0, 1))
                for v2 in settings.weight_combination_list:
                    hyperparams_evaluation_list.append((v1,) + v2)

            for feat_strategy in settings.num_feat_strategy:
                tmp_paths = train_path_recommender(data_log=data_log, train_val_log=train_val_log, val_log=val_log,
                                                   train_log=train_log, labeling=labeling,
                                                   support_threshold=settings.support_threshold_dict,
                                                   checkers=settings.checkers[constr_family], rules=settings.rules,
                                                   dataset_name=dataset_name, constr_family=constr_family,
                                                   output_dir=settings.output_dir, min_prefix_length=min_prefix_length,
                                                   max_prefix_length=max_prefix_length_test,
                                                   feat_strategy=feat_strategy)
                feat_strategy_paths_dict[feat_strategy] = tmp_paths

                # discovering on val set with best hyperparams_evaluation setting
                print(f"Hyper params for evaluation for {dataset_name} ...")
                if compute_baseline:
                    hyperparams_evaluation_list = hyperparams_evaluation_list_baseline

                for hyperparams_evaluation in hyperparams_evaluation_list:
                    res_val_list = []
                    eval_res = None
                    if settings.cumulative_res is True:
                        eval_res = EvaluationResult()
                    for pref_id, prefix_len in enumerate(prefix_lenght_list_val):
                        prefixing = {
                            "type": PrefixType.ONLY,
                            "length": prefix_len
                        }
                        recommendations, evaluation, _ = generate_recommendations_and_evaluation(test_log=val_log,
                                                                                                 train_log=train_log,
                                                                                                 labeling=labeling,
                                                                                                 prefixing=prefixing,
                                                                                                 support_threshold=settings.support_threshold_dict,
                                                                                                 checkers=settings.checkers[constr_family],
                                                                                                 rules=settings.rules,
                                                                                                 paths=tmp_paths,
                                                                                                 dataset_name=dataset_name,
                                                                                                 hyperparams_evaluation=hyperparams_evaluation,
                                                                                                 eval_res=eval_res,
                                                                                                 constr_family=constr_family)
                        if settings.cumulative_res is True:
                            eval_res = copy.deepcopy(evaluation)
                        res_val_list.append(eval_res.fscore)
                    results_hyperparams_evaluation[(feat_strategy, ) + hyperparams_evaluation] = np.mean(res_val_list)

            results_hyperparams_evaluation = dict(sorted(results_hyperparams_evaluation.items(), key=lambda item: item[1]))
            best_hyperparams_combination = list(results_hyperparams_evaluation.keys())[-1]
            paths = feat_strategy_paths_dict[best_hyperparams_combination[0]]
            best_hyperparams_combination = best_hyperparams_combination[1:]
            print(f"BEST HYPERPARAMS COMBINATION {best_hyperparams_combination}")
            with open(os.path.join(settings.models_path, f'{dataset_name}_{constr_family}.pickle'), 'wb') as file:
                pickle.dump((paths, best_hyperparams_combination), file)

        # saving training/optimization times
        training_optimization_time = timeit.default_timer() - start_train_opt_time
        train_opt += [[dataset_name, constr_family, training_optimization_time]]

        # testing on test set with best hyperparams_evaluation setting
        eval_res = None
        if settings.cumulative_res is True:
            eval_res = EvaluationResult()

        for pref_id, prefix_len in enumerate(prefix_lenght_list_test):
            print(
                f"<--- DATASET: {dataset_name}, CONSTRAINTS: {constr_family},"
                f"PREFIX LEN: {prefix_len}/{max_prefix_length_test} --->")
            prefixing = {
                "type": PrefixType.ONLY,
                "length": prefix_len
            }
            recommendations, evaluation, time_results_tmp = generate_recommendations_and_evaluation(test_log=test_log,
                                                                                                    train_log=train_log,
                                                                                                    labeling=labeling,
                                                                                                    prefixing=prefixing,
                                                                                                    support_threshold=settings.support_threshold_dict,
                                                                                                    checkers=settings.checkers[constr_family],
                                                                                                    rules=settings.rules,
                                                                                                    paths=paths,
                                                                                                    dataset_name=dataset_name,
                                                                                                    hyperparams_evaluation=best_hyperparams_combination,
                                                                                                    eval_res=eval_res,
                                                                                                    constr_family=constr_family,
                                                                                                    debug=False)
            results[constr_family].append(evaluation)
            time_results += time_results_tmp
            if settings.cumulative_res is True:
                eval_res = copy.deepcopy(evaluation)

            for metric in ["fscore"]:  # ["accuracy", "fscore", "auc", "gain"]:
                print(f"{metric} for {constr_family}: {getattr(results[constr_family][pref_id], metric)}")

    recommendation_times_file = os.path.join(settings.results_dir, f"{dataset_name}_recommendation_times.csv")
    with open(recommendation_times_file, 'w') as f:
        writer = csv.writer(f)
        for res in time_results:
            writer.writerow(res)

    train_opt_times_file = os.path.join(settings.results_dir, f"{dataset_name}_train_opt_times.csv")
    with open(train_opt_times_file, 'w') as f:
        writer = csv.writer(f)
        for res in train_opt:
            writer.writerow(res)
    plot = PlotResult(results, prefix_lenght_list_test, settings.results_dir)

    for metric in ["fscore"]:
        plot.toPng(metric, f"{dataset_name}_{metric}")
        """
        with open(os.path.join(results_dir, f"{dataset_name}_{metric}.csv"), mode='w') as out_file:
            writer = csv.writer(out_file, delimiter=',')
            for constr_family in constr_family_list:
                writer.writerow([constr_family] + [getattr(res_obj, metric) for res_obj in results[constr_family]])
        """
    prefix_evaluation_to_csv(results, dataset_name)
    return dataset_name, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments for outcome-based prescriptive process monitoring")
    parser.add_argument('--log', default=None, help='input log')
    parser.add_argument('--load_model', action='store_true', help='Use trained model')
    parser.add_argument('--baseline', action='store_true', help='Use baseline model')
    parser.add_argument('--decl_list', help='delimited list input', type=str)

    args = parser.parse_args()
    dataset = args.log
    load_model = args.load_model
    compute_baseline = args.baseline
    input_constraint_families = args.decl_list
    if input_constraint_families is not None:
        settings.constr_family_list = []
        for constraint_family in input_constraint_families.split(','):
            if constraint_family in settings.checkers.keys():
                settings.constr_family_list.append(constraint_family)
            else:
                raise Exception(f"{constraint_family} not allowed. Allowed DECLARE families are existence, choice,"
                                f"positive relations, negative relations, all.")

    start_time = time.time()
    _, res_obj = rec_sys_exp(dataset)
    print(f"Simulations took {(time.time() - start_time) / 3600.} hours for {dataset}")