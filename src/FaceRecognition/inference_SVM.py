"""
    arcface extract feature + linearSVC classifier
"""

import numpy as np


def inference_identity(embeded, model, threshold = -0.55):

    pred_label = model.predict([embeded])[0]
    # pred_funct = model.decision_function(embeded)


    return pred_label
    # funct = list(pred_funct)
    # index = list(np.arange(5))

    # funct_plus = [threshold]
    # funct_plus.extend(list(pred_funct))
    # index_plus = list(np.arange(5))
    

    # funct_plus, index_plus = zip(*sorted(zip(funct_plus, index_plus), key = lambda x: -x[0]))
    # funct, index = zip(*sorted(zip(funct, index), key = lambda x: -x[0]))