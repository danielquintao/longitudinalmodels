import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_lcga_TWO_groups(betas, time, data, degree, responsibility):
    """plots dataset growth curves along with GCM curve per group.
       The number of groups in inferred from the arguments

    Args:
        betas (pair (list or tuple) of (degree+1,1) numpy arrays): fixed effects of each latent class
        time (numpy array of length T): [time points (supposed to be the same for all individuals)]
        data (2D numpy array of shape (N,T)): [time-observations for the N individuals]
        degree (int): degree of the polynomial
        responsibility (callable): function that evaluates the chances of observation yi belonging to each class
    """
    assert len(betas) == 2
    assert all([beta.shape == (degree+1,1) for beta in betas])
    N,T = data.shape
    assert T == len(time)
    plt.figure()
    # plot individual, observed curves
    for i in range(N):
        # do not plot people that belong to groups that are not present in groups2plot:
        # if tuple(groups[i]) not in groups2plot:
        #     continue
        color = plt.cm.winter(256*(1-responsibility(data[i].reshape(-1,1))))[0,0]
        plt.plot(time, data[i], color=color, linestyle='dotted', linewidth=1)
    # plot population-level curves
    for _, beta in enumerate(betas):
        curve = np.zeros(T)
        for i in range(degree+1):
            curve += beta[i] * time**i
        plt.plot(time, curve, color=plt.cm.winter(256*_), linewidth=5)
    # legends
    plt.show()
