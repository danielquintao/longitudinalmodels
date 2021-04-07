import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(vector_eta, time, data, degree):
    """plots dataset growth curves along with GCM curve

    Args:
        vector_eta (1D ndarray of shape (2,)): [intercept and slope resp.]
        time (list or ndarray of length T): [time points (spposed the same for all individuals)]
        data (2D ndarray of shape (N,T)): [time-observations for the N individuals]
    """
    assert vector_eta.shape == (degree+1,)
    N,T = data.shape
    assert T == len(time)
    curve = vector_eta[0] # + vector_eta[1] * time
    for i in range(1, degree+1):
        curve += vector_eta[i] * (time ** i)
    plt.figure()
    for i in range(N):
        plt.plot(time, data[i], linewidth=1)
    plt.plot(time, curve, 'k-', linewidth=5)
    plt.show()