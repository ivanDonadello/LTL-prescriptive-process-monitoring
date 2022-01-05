import copy

import pm4py
import os
import shutil
import csv
import pdb
import settings
import platform
from src.dataset_manager.datasetManager import DatasetManager
from src.enums.ConstraintChecker import ConstraintChecker
from src.machine_learning.utils import *
from src.machine_learning import *
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
import argparse
import multiprocessing
import sys
import time
import numpy as np


def rec_sys_exp(dataset_name):

    # ================ inputs ================

    # recreate ouput folder
    # shutil.rmtree("media/output", ignore_errors=True)
    # os.makedirs(os.path.join(results_dir))

    # generate rules
    settings.rules["activation"] = generate_rules(settings.rules["activation"])
    settings.rules["correlation"] = generate_rules(settings.rules["correlation"])

    dataset_manager = DatasetManager(dataset_name.lower())
    data = dataset_manager.read_dataset(os.path.join(os.getcwd(), settings.dataset_folder))
    #"""

    #"""

    # split into training and test
    train_val_ratio = 0.8
    train_ratio = 0.9
    train_val_df, test_df = dataset_manager.split_data_strict(data, train_val_ratio, split="temporal")
    train_df, val_df = dataset_manager.split_data_strict(train_val_df, train_ratio, split="temporal")

    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length = 9
    elif "bpic2017" in dataset_name:
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(test_df, 0.90))
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(test_df, 0.90))

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


    # TODO trace bucketing
    # train_log_al = pm4py.read_xes(train_log_path)
    # test_log_al = pm4py.read_xes(test_log_path)

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
        "type": LabelType.TRACE_DURATION,
        "threshold_type": LabelThresholdType.LABEL_MEAN,
        "target": TraceLabel.TRUE,  # lower than a threshold considered as True
        "trace_attribute": "",
        "custom_threshold": 0.0
    }
    """

    # generate recommendations and evaluation
    results = {family: [] for family in settings.constr_family_list}
    for constr_family in settings.constr_family_list:
        paths = train_path_recommender(data_log=data_log, train_val_log=train_val_log, val_log=val_log, train_log=train_log, labeling=labeling, support_threshold=settings.support_threshold_dict,
                                       checkers=settings.checkers[constr_family], rules=settings.rules, dataset_name=dataset_name, constr_family=constr_family,
                                       output_dir=settings.output_dir, min_prefix_length=min_prefix_length, max_prefix_length=max_prefix_length)
        prefix_lenght_list = list(range(min_prefix_length, max_prefix_length + 1))

        eval_res = None
        if settings.cumulative_res is True:
            eval_res = EvaluationResult()

        for pref_id, prefix_len in enumerate(prefix_lenght_list):
            print(
                f"<--- DATASET: {dataset_name}, CONSTRAINTS: {constr_family}, PREFIX LEN: {prefix_len}/{max_prefix_length} --->")
            prefixing = {
                "type": PrefixType.ONLY,
                "length": prefix_len
            }
            recommendations, evaluation = generate_recommendations_and_evaluation(test_log=test_log,
                                                                                  train_log=train_log,
                                                                                  labeling=labeling,
                                                                                  prefixing=prefixing,
                                                                                  support_threshold=settings.support_threshold_dict,
                                                                                  checkers=settings.checkers[constr_family],
                                                                                  rules=settings.rules,
                                                                                  paths=paths,
                                                                                  dataset_name=dataset_name,
                                                                                  eval_res=eval_res)
            results[constr_family].append(evaluation)
            if settings.cumulative_res is True:
                eval_res = copy.deepcopy(evaluation)

            for metric in ["fscore", "gain"]:  # ["accuracy", "fscore", "auc"]:
                print(f"{metric} for {constr_family}: {getattr(results[constr_family][pref_id], metric)}")
    plot = PlotResult(results, prefix_lenght_list, settings.results_dir)

    for metric in ["fscore", "gain"]:
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
        print_lock = multiprocessing.Lock()
        parser = argparse.ArgumentParser(description="Experiments for outcome-based prescriptive process monitoring")
        parser.add_argument("-j", "--jobs", type=int, help="Number of jobs to run in parallel. If -1 all CPUs are used.")
        args = parser.parse_args()

        jobs = None
        available_jobs = multiprocessing.cpu_count()
        if args.jobs:
            if args.jobs < -1 or args.jobs == 0:
                print(f"-j must be -1 or grater than 0")
                sys.exit(2)
            jobs = available_jobs if args.jobs == -1 else args.jobs

        final_results = {}
        start_time = time.time()
        if jobs is None or jobs == 1:
            for dataset in settings.datasets_names:
                _, res_obj = rec_sys_exp(dataset)
                final_results[dataset] = res_obj
        else:
            tmp_list_results = []
            if platform.platform().split('-')[0] == 'macOS' or platform.platform().split('-')[0] == 'Darwin':
                with multiprocessing.get_context("spawn").Pool(processes=jobs) as pool:
                    tmp_list_results = pool.map(rec_sys_exp, settings.datasets_names)
            else:
                pool = multiprocessing.Pool(processes=jobs)
                # tmp_list_results = [pool.apply_async(rec_sys_exp, args=(ds, )) for ds in settings.datasets_names]
                # pool.close()

                tmp_list_results = pool.map(rec_sys_exp, settings.datasets_names)
                pool.close()

                """
                r = []
                for p in tmp_list_results:
                    try:
                        r.append(p.get())
                    except Exception:
                        print("error getting process: %s" % os.getpid())
                r = tmp_list_results
                """
            final_results = dict(tmp_list_results)

        with open(os.path.join(settings.output_dir, f"results.csv"), mode='w') as out_file:
            writer = csv.writer(out_file, delimiter=',')
            writer.writerow(["Dataset"] + 2*list(settings.constr_family_list))
            for dataset in settings.datasets_names:
                writer.writerow([dataset] +
                                [round(100*np.mean([getattr(res_obj, 'fscore') for res_obj in final_results[dataset][constr_family]]), 2) for constr_family in settings.constr_family_list] +
                                [round(np.mean([getattr(res_obj, 'gain') for res_obj in final_results[dataset][constr_family]]), 2) for constr_family in settings.constr_family_list])
        print(f"Simulations took {(time.time() - start_time) / 3600.} hours")