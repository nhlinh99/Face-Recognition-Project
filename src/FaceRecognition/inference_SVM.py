"""
    arcface extract feature + linearSVC classifier
"""

import numpy as np
import os

base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
dataset_path = base_path + "/dataset/label/"
labels = ['0', '1', '2', '3']

def inference_identity(embeded, model, threshold = -0.55):

    # pred_label = model.predict([embeded])
    pred_funct = model.decision_function([embeded])

    score_prediction = pred_funct.tolist()[0]

    index = range(len(os.listdir(dataset_path)))

    zip_id_score_prediction = sorted(list(zip(index, score_prediction)), key = lambda x: -x[1])

    if (zip_id_score_prediction[0][0] == 0) or (zip_id_score_prediction[0][1] <= threshold):
        return "Unknown"
    else:
        return labels[zip_id_score_prediction[0][0]]
