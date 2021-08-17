import pm4py
import os
import shutil
import csv
import pdb
from src.dataset_manager.datasetManager import DatasetManager
from src.enums.ConstraintChecker import ConstraintChecker
from src.machine_learning.utils import *
from src.machine_learning import *
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter


if __name__ == "__main__":
        # ================ inputs ================
        support_threshold = 0.75
        output_dir = "media/output/result"
        train_log_path = "media/input/log/train.xes"
        test_log_path = "media/input/log/test.xes"
        dataset_folder = "media/input/processed_benchmark_event_logs"
        datasets_names = ["bpic2011_f1", "bpic2011_f2", "bpic2011_f3", "bpic2011_f4",
                          "bpic2015_1_f2", "bpic2015_2_f2", "bpic2015_3_f2", "bpic2015_4_f2",
                          "bpic2015_5_f2", "bpic2017_accepted", "bpic2017_cancelled",
                          "bpic2017_refused", "bpic2012_cancelled",
                          "bpic2012_accepted", "bpic2012_declined",
                          "hospital_billing_2", "hospital_billing_3", "Production",
                          "sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4", "traffic_fines_1"]
        datasets_names = ["sepsis_cases_1"]

        checkers = {"existence": [ConstraintChecker.EXISTENCE,
            ConstraintChecker.ABSENCE,
            ConstraintChecker.INIT,
            ConstraintChecker.EXACTLY],
                    "choice": [ConstraintChecker.EXISTENCE,
            ConstraintChecker.ABSENCE,
            ConstraintChecker.INIT,
            ConstraintChecker.EXACTLY,
            ConstraintChecker.CHOICE,
            ConstraintChecker.EXCLUSIVE_CHOICE],
                    "positive relations": [ConstraintChecker.EXISTENCE,
            ConstraintChecker.ABSENCE,
            ConstraintChecker.INIT,
            ConstraintChecker.EXACTLY,
            ConstraintChecker.CHOICE,
            ConstraintChecker.EXCLUSIVE_CHOICE,
            ConstraintChecker.RESPONDED_EXISTENCE,
            ConstraintChecker.RESPONSE,
            ConstraintChecker.ALTERNATE_RESPONSE,
            ConstraintChecker.CHAIN_RESPONSE,
            ConstraintChecker.PRECEDENCE,
            ConstraintChecker.ALTERNATE_PRECEDENCE,
            ConstraintChecker.CHAIN_PRECEDENCE],
                    "negative relations": [ConstraintChecker.EXISTENCE,
            ConstraintChecker.ABSENCE,
            ConstraintChecker.INIT,
            ConstraintChecker.EXACTLY,
            ConstraintChecker.CHOICE,
            ConstraintChecker.EXCLUSIVE_CHOICE,
            ConstraintChecker.RESPONDED_EXISTENCE,
            ConstraintChecker.RESPONSE,
            ConstraintChecker.ALTERNATE_RESPONSE,
            ConstraintChecker.CHAIN_RESPONSE,
            ConstraintChecker.PRECEDENCE,
            ConstraintChecker.ALTERNATE_PRECEDENCE,
            ConstraintChecker.CHAIN_PRECEDENCE,
            ConstraintChecker.NOT_RESPONDED_EXISTENCE,
            ConstraintChecker.NOT_RESPONSE,
            ConstraintChecker.NOT_CHAIN_RESPONSE,
            ConstraintChecker.NOT_PRECEDENCE,
            ConstraintChecker.NOT_CHAIN_PRECEDENCE]}

        constr_family_list = ["existence", "choice", "positive relations", "negative relations"]#checkers.keys()

        rules = {
            "vacuous_satisfaction": True,
            "activation": "", # e.g. A.attr > 6
            "correlation": "", # e.g. T.attr < 12
            "n": {
                ConstraintChecker.EXISTENCE: 1,
                ConstraintChecker.ABSENCE: 1,
                ConstraintChecker.EXACTLY: 1,
            }
        }

        # ================ inputs ================

        # recreate ouput folder
        shutil.rmtree("media/output", ignore_errors=True)
        os.makedirs(os.path.join(output_dir))

        # generate rules
        rules["activation"] = generate_rules(rules["activation"])
        rules["correlation"] = generate_rules(rules["correlation"])

        # read the datasets
        for dataset_name in datasets_names:

            dataset_manager = DatasetManager(dataset_name.lower())
            data = dataset_manager.read_dataset(os.path.join(os.getcwd(), dataset_folder))

            # determine min and max (truncated) prefix lengths
            min_prefix_length = 1
            if "traffic_fines" in dataset_name:
                max_prefix_length = 10
            elif "bpic2017" in dataset_name:
                max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
            else:
                max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

            # split into training and test
            train_ratio = 0.8
            train_df, test_df = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
            train_df = train_df.rename(columns={dataset_manager.timestamp_col: 'time:timestamp', dataset_manager.case_id_col: 'case:concept:name', dataset_manager.activity_col: 'concept:name'})
            test_df = test_df.rename(columns={dataset_manager.timestamp_col: 'time:timestamp', dataset_manager.case_id_col: 'case:concept:name', dataset_manager.activity_col: 'concept:name'})
            train_log = log_converter.apply(train_df)
            test_log = log_converter.apply(test_df)

            # TODO trace bucketing
            # TODO iperparametri

            #train_log_al = pm4py.read_xes(train_log_path)
            #test_log_al = pm4py.read_xes(test_log_path)

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
            results = {family: [] for family in constr_family_list}
            for constr_family in constr_family_list:
                for pref_id, prefix_len in enumerate(range(min_prefix_length, max_prefix_length+1)):
                    print(f"<--- DATASET: {dataset_name}, CONSTRAINTS: {constr_family}, PREFIX LEN: {prefix_len}/{max_prefix_length} --->")
                    prefixing = {
                        "type": PrefixType.ONLY,
                        "length": prefix_len
                    }
                    recommendations, evaluation = generate_recommendations_and_evaluation(test_log=test_log, train_log=train_log,
                                                                                  labeling=labeling,
                                                                                  prefixing=prefixing,
                                                                                  support_threshold=support_threshold,
                                                                                  checkers=checkers[constr_family],
                                                                                  rules=rules)
                    results[constr_family].append(evaluation)

                    for metric in ["accuracy", "fscore", "auc"]:
                        print(f"{metric} for {constr_family}: {getattr(results[constr_family][pref_id], metric)}")
            plot = PlotResult(results, folder=output_dir)

            for metric in ["accuracy", "fscore", "auc"]:
                plot.toPng(f"{metric}", f"{dataset_name}_{metric}")
                with open(os.path.join(output_dir, f"{dataset_name}_{metric}.csv"), mode='w') as out_file:
                    writer = csv.writer(out_file, delimiter=',')
                    for constr_family in constr_family_list:
                        writer.writerow([constr_family] + [getattr(res_obj, metric) for res_obj in results[constr_family]])