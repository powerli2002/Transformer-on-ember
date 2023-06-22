import os
import lief
import numpy as np
import ember
from utils import predict_sample
import matplotlib.pyplot as plt

from classify_DL import Classifier_DL
import lightgbm as lgb

def find_exe_files(folder_path):
    exe_files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.exe'):
                exe_files.append(os.path.join(dirpath, filename))
    return exe_files

def evaluate_detection(folder_path):
    total = 0
    DL_mal = 0
    DL_good = 0
    ember_mal = 0
    ember_good = 0
    ensemble_mal = 0
    ensemble_good = 0

    def predict_detection(model, file_data):
        score = predict_sample(model, file_data, 2)
        return score

    exe_files = find_exe_files(folder_path)

    ember_scores = []
    transformer_scores = []
    ensemble_scores = []

    modelpath = "/home/lizijian/lincode/ember/data/ember2018/ember_model_2018.txt"
    lgbm_model = lgb.Booster(model_file=modelpath)

    classify_DL = Classifier_DL("histogram")
    total = len(exe_files)

    results = []

    for exe_file in exe_files:
        DL_ans, DL_output = classify_DL.detect_file(exe_file)

        file_data = open(exe_file, "rb").read()
        score = predict_detection(lgbm_model, file_data)

        ensemble_ans = 1 if score > 0.4 or DL_ans == 1 else 0

        result = {
            "file_name": exe_file,
            "DL_ans": DL_ans,
            "DL_output": DL_output.data,
            "ember_score":1 if score > 0.4 else 0,
            "ensemble_ans": ensemble_ans
        }
        results.append(result)

    # Calculate statistics
    for result in results:
        DL_ans = result["DL_ans"]
        DL_output = result["DL_output"]
        ember_score = result["ember_score"]
        ensemble_ans = result["ensemble_ans"]

        if DL_ans == 1:
            DL_mal += 1
        else:
            DL_good += 1

        if ember_score > 0.4:
            ember_mal += 1
        else:
            ember_good += 1

        if ensemble_ans == 1:
            ensemble_mal += 1
        else:
            ensemble_good += 1

    statistics = {
        "total": total,
        "DL_malicious": DL_mal,
        "DL_detection_rate": DL_mal / total,
        "ember_malicious": ember_mal,
        "ember_detection_rate": ember_mal / total,
        "ensemble_malicious": ensemble_mal,
        "ensemble_detection_rate": ensemble_mal / total
    }

    return results, statistics
