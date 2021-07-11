from declare_based.src.enums.ConstraintChecker import ConstraintChecker
from django.views.decorators.csrf import csrf_exempt
from declare_based.src.parsers import *
from declare_based.src.machine_learning import *
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from pm4py.objects.log.importer.xes import factory as xes_import_factory

import os
import shutil

class DeclareTemplates:
    @csrf_exempt
    @api_view(['GET'])
    def generate(request):
        log = "test.xes" # name of the .xes file
        declare = "test.decl" # name of the .decl file
        done = True # whether the log is complete or not

        log = xes_import_factory.apply(settings.MEDIA_ROOT + "input/log/" + log)
        declare = parse_decl(settings.MEDIA_ROOT + "input/declare/" + declare)

        activities = declare["activities"]
        result = {}
        for key, rules in declare["rules"].items():
            if key == ConstraintChecker.INIT.value:
                result[key] = DT_LOG_METHODS[key](log, done, activities["A"], rules["activation_rules"]).__dict__
            elif key in [ConstraintChecker.EXISTENCE.value, ConstraintChecker.ABSENCE.value, ConstraintChecker.EXACTLY.value]:
                result[key] = DT_LOG_METHODS[key](log, done, activities["A"], rules["activation_rules"],
                                                  rules["n"]).__dict__
            elif key in [ConstraintChecker.CHOICE.value, ConstraintChecker.EXCLUSIVE_CHOICE.value]:
                result[key] = DT_LOG_METHODS[key](log, done, activities["A"], activities["T"],
                                                  rules["activation_rules"]).__dict__
            else:
                result[key] = DT_LOG_METHODS[key](log, done, activities["A"], activities["T"],
                                                  rules["activation_rules"],
                                                  rules["correlation_rules"]).__dict__
        context = {"result": result}
        return Response(context, status=status.HTTP_200_OK)

class Recommendation:
    @csrf_exempt
    @api_view(['GET'])
    def recommend(request):
        labeling = {
            "label_type": LabelType.TRACE_DURATION,
            "label_threshold_type": LabelThresholdType.LABEL_MEAN,
            "target_label": TraceLabel.TRUE,
            "trace_attribute": "",
            "custom_label_threshold": 0.0
        }
        prefix_type = {
            "type": PrefixType.UPTO,
            "length": 5
        }
        templates = [
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
        rules = {
            "vacuous_satisfaction": True,
            "activation": "",
            "correlation": ""
        }
        support_threshold = 0.75

        rules["activation"] = generate_rules(rules["activation"])
        rules["correlation"] = generate_rules(rules["correlation"])

        shutil.rmtree(settings.MEDIA_ROOT + "output", ignore_errors=True)
        os.makedirs(os.path.join(settings.MEDIA_ROOT + "output/result"))
        
        train_log = xes_import_factory.apply(settings.MEDIA_ROOT + "input/log/train.xes")
        test_log = xes_import_factory.apply(settings.MEDIA_ROOT + "input/log/test.xes")

        recommendations, evaluation = generate_recommendations_and_evaluation(test_log=test_log, train_log=train_log,
                                                                              labeling=labeling,
                                                                              prefix_type=prefix_type,
                                                                              support_threshold=support_threshold,
                                                                              templates=templates,
                                                                              rules=rules)
        res = []
        for recommendation in recommendations:
            res.append(recommendation.__dict__)
        context = {"recommendations": res, "evaluation": evaluation.__dict__}
        return Response(context, status=status.HTTP_200_OK)