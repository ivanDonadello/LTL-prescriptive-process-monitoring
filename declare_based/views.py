from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from declare_based.src.parsers import *
from declare_based.src.machine_learning import *
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from pm4py.objects.log.importer.xes import factory as xes_import_factory

import os
import json
import shutil
import requests


class DeclareTemplates:
    @csrf_exempt
    @api_view(['POST'])
    def generate(request):
        log = request.data.get("log", None)
        declare = request.data.get("declare", None)
        done = json.loads(request.data.get('done', None))

        log_path = settings.MEDIA_ROOT + "input/log/" + log
        declare_path = settings.MEDIA_ROOT + "input/declare/" + declare

        log = xes_import_factory.apply(log_path)
        declare = parse_decl(declare_path)

        activities = declare["activities"]
        result = {}
        for key, rules in declare["rules"].items():
            if key == INIT:
                result[key] = DT_LOG_METHODS[key](log, done, activities["A"], rules["activation_rules"]).__dict__
            elif key in [EXISTENCE, ABSENCE, EXACTLY]:
                result[key] = DT_LOG_METHODS[key](log, done, activities["A"], rules["activation_rules"],
                                                  rules["n"]).__dict__
            elif key in [CHOICE, EXCLUSIVE_CHOICE]:
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
    @api_view(['POST'])
    def recommend(request):

        selected_log_split_id = request.data.get("selectedLogSplitId", None)
        labeling = request.data.get("labeling", None)
        prefix_type = request.data.get("prefix", None)
        thresholds = request.data.get("thresholds", None)
        templates = request.data.get("templates", None)
        rules = request.data.get("rules", None)
        support_threshold = thresholds["supportThreshold"]

        rules["activation"] = generate_rules(rules["activation"])
        rules["correlation"] = generate_rules(rules["correlation"])

        train_url = "http://193.40.11.150/splits/" + str(selected_log_split_id) + "/logs/train"
        test_url = "http://193.40.11.150/splits/" + str(selected_log_split_id) + "/logs/test"
        train_data = requests.get(train_url, allow_redirects=True)
        test_data = requests.get(test_url, allow_redirects=True)

        if os.path.exists(settings.MEDIA_ROOT + "input/log/splits"):
            shutil.rmtree(settings.MEDIA_ROOT + "input/log/splits", ignore_errors=True)
        os.makedirs(os.path.join(settings.MEDIA_ROOT + "input/log/splits/" + str(selected_log_split_id) + "/"))
        train_path = settings.MEDIA_ROOT + "input/log/splits/" + str(selected_log_split_id) + "/train.xes"
        test_path = settings.MEDIA_ROOT + "input/log/splits/" + str(selected_log_split_id) + "/test.xes"

        open(train_path, 'wb').write(train_data.content)
        open(test_path, 'wb').write(test_data.content)

        train_log = xes_import_factory.apply(train_path)
        test_log = xes_import_factory.apply(test_path)

        recommendations, evaluation = generate_recommendations_and_evaluation(test_log=test_log, train_log=train_log,
                                                                              labeling=labeling,
                                                                              prefix_type=prefix_type,
                                                                              support_threshold=float(
                                                                                  support_threshold),
                                                                              templates=templates,
                                                                              rules=rules)
        res = []
        for recommendation in recommendations:
            res.append(recommendation.__dict__)
        context = {"recommendations": res, "evaluation": evaluation.__dict__}
        return Response(context, status=status.HTTP_200_OK)
