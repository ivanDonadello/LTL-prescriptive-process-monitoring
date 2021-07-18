from fastapi import FastAPI

from pm4py.objects.log.importer.xes import factory as xes_import_factory

import os
import shutil

from src.enums.ConstraintChecker import ConstraintChecker
from src.machine_learning.utils import *
from src.machine_learning import *

app = FastAPI()

@app.get("/")
def recommend():
        # ================ inputs ================
        support_threshold = 0.75
        train_log_path = "media/input/log/train.xes"
        test_log_path = "media/input/log/test.xes"
        checkers = [
            ConstraintChecker.EXISTENCE,
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
            ConstraintChecker.NOT_CHAIN_PRECEDENCE
        ]
        prefixing = {
            "type": PrefixType.UPTO,
            "length": 5
        }
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
        labeling = {
            "type": LabelType.TRACE_DURATION,
            "threshold_type": LabelThresholdType.LABEL_MEAN,
            "target": TraceLabel.TRUE, # lower than a threshold considered as True
            "trace_attribute": "",
            "custom_threshold": 0.0
        }
        # ================ inputs ================

        # recreate ouput folder
        shutil.rmtree("media/output", ignore_errors=True)
        os.makedirs(os.path.join("media/output/result"))

        # generate rules
        rules["activation"] = generate_rules(rules["activation"])
        rules["correlation"] = generate_rules(rules["correlation"])

        # read the files
        train_log = xes_import_factory.apply(train_log_path)
        test_log = xes_import_factory.apply(test_log_path)

        # generate recommendations and evaluation
        recommendations, evaluation = generate_recommendations_and_evaluation(test_log=test_log, train_log=train_log,
                                                                              labeling=labeling,
                                                                              prefixing=prefixing,
                                                                              support_threshold=support_threshold,
                                                                              checkers=checkers,
                                                                              rules=rules)
                                                                              
        return {"recommendations": recommendations, "evaluation": evaluation}