import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    """
    sigmoid function

    params:
    z:
    

    returns:
    probability
    """

    return 1.0 / (1.0 + np.exp(-z))