import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot(vector_eta, time, data, degree):
    """plots dataset growth curves along with GCM curve

    Args:
        vector_eta (1D ndarray of length degree+1): curve coefficients
        time (list or numpy array of length T): [time points (supposed to be the same for all individuals)]
        data (2D ndarray of shape (N,T)): [time-observations for the N individuals]
        degree (int): degree of the polynomial
    """
    assert vector_eta.shape == (degree+1,)
    N,T = data.shape
    assert T == len(time)
    curve = vector_eta[0]
    for i in range(1, degree+1):
        curve += vector_eta[i] * (time ** i)
    plt.figure()
    for i in range(N):
        plt.plot(time, data[i], linewidth=1)
    plt.plot(time, curve, 'k-', linewidth=5)
    plt.show()

def extended_plot(vector_eta, time, data, groups, groups2plot, degree):
    """plots dataset growth curves along with GCM curve per group.
       The number of groups in inferred from the arguments

    Args:
        vector_eta (1D numpy array of length (M+1)*(degree+1)): The first (degree+1) vlues are the 'central' curve coefficients
                                                                and the following batches of degree+1 consecutive values are the
                                                                deviations we need to add to have the coefficients for the groups
                                                                (e.g. if there are 2 groups encoded by 0 and 1, and degree is 1, then
                                                                the first 2 values are the curve a0+a1*t for group 0, while group 1 is
                                                                represented by (a0+a2) + (a1+a3)*x)
        time (list or numpy array of length T): [time points (supposed to be the same for all individuals)]
        data (2D numpy array of shape (N,T)): [time-observations for the N individuals]
        groups (2D numpy array of shape (N,M)): each group is encoded by a combination of M binary values 
        groups2plot (list of tuples): combinations of our M binary variables that should be plotted as a group. For example, we may
                                      have groups2plot = [(0,0), (1,0), (0,1)] for 3 groups represented by 2 variables.
                                      There is support to plot at most 10 groups, but code can be adapted to plot more.
        degree (int): degree of the polynomial
    """
    n_grouping_vars = groups.shape[1]
    assert vector_eta.shape == ((n_grouping_vars+1)*(degree+1),)
    N,T = data.shape
    assert T == len(time)
    assert groups.shape[0] == N
    assert all([len(x) == n_grouping_vars for x in groups2plot])
    colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple',
     5:'tab:brown', 6:'tab:pink', 7:'tab:gray', 8:'tab:olive', 9:'tab:cyan'}
    plt.figure()
    # plot individual, observed curves
    for i in range(N):
        # do not plot people that belong to groups that are not present in groups2plot:
        # if tuple(groups[i]) not in groups2plot:
        #     continue
        color = colors[groups2plot.index(tuple(groups[i]))]
        plt.plot(time, data[i], color=color, linestyle='dotted', linewidth=1)
    # plot population-level curves
    for counter, g in enumerate(groups2plot):
        curve = np.zeros(T)
        coeffs = vector_eta[0:degree+1]
        for i, bin_var in enumerate(g,start=1):
            coeffs += bin_var * vector_eta[i*(degree+1) : (i+1)*(degree+1)]
        for i in range(degree+1):
            curve += coeffs[i] * time**i
        plt.plot(time, curve, color=colors[counter], linewidth=5)
    # legends
    legend = ['group '+str(x) for x in groups2plot]
    handles = [Line2D([0],[0],color=colors[i]) for i in range(len(groups2plot))]
    plt.legend(handles, legend)
    plt.show()
