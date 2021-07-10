from declare_based.src.enums.ConstraintChecker import ConstraintChecker
from django.views.decorators.csrf import csrf_exempt
from declare_based.src.parsers import *
from declare_based.src.machine_learning import *
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from pm4py.objects.log.importer.xes import factory as xes_import_factory

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
        
        train_log = xes_import_factory.apply(settings.MEDIA_ROOT + "input/log-splits/train.xes")
        test_log = xes_import_factory.apply(settings.MEDIA_ROOT + "input/log-splits/test.xes")

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