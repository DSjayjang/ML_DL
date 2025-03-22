import numpy as np

def accuracy(y_true, y_pred):
    """
    metrics: accuracy

    params:
    y_true:
    y_pred: 

    returns:
    accuracy:
    """

    accuracy = np.sum(y_true == y_pred) / len(y_true)

    return accuracy