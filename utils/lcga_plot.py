import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_lcga_TWO_groups(betas, time, data, degree, responsibility):
    """plots dataset with two clusters and color corresponding to the subject's prob of belonging to cluster.

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

def plot_lcga(betas, time, data, clusters_pred, degree):
    """plots dataset with two clusters and color corresponding to the subject's prob of belonging to cluster.

    Args:
        betas (pair (list or tuple) of (degree+1,1) numpy arrays): fixed effects of each latent class
        time (numpy array of length T): [time points (supposed to be the same for all individuals)]
        data (2D numpy array of shape (N,T)): [time-observations for the N individuals]
        clusters_pred (1D array of type int): most likely cluster of each individual
        degree (int): degree of the polynomial
    """
    # we'll take advantage of the similarity between our problem and to plot a GCM estimation with classes
    # and have a very similar function, differing on the data format
    n_clusters = max(clusters_pred)+1
    assert len(betas) == n_clusters
    N,T = data.shape
    assert T == len(time)
    assert len(clusters_pred) == N
    colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple',
     5:'tab:brown', 6:'tab:pink', 7:'tab:gray', 8:'tab:olive', 9:'tab:cyan'}
    plt.figure()
    # plot individual, observed curves
    for i in range(N):
        # do not plot people that belong to groups that are not present in groups2plot:
        # if tuple(groups[i]) not in groups2plot:
        #     continue
        color = colors[clusters_pred[i]]
        plt.plot(time, data[i], color=color, linestyle='dotted', linewidth=1)
    # plot population-level curves
    for counter in range(n_clusters):
        curve = np.zeros(T)
        coeffs = np.copy(betas[counter])
        for i in range(degree+1):
            curve += coeffs[i] * time**i
        plt.plot(time, curve, color=colors[counter], linewidth=5)
    # legends
    legend = ['group '+str(x) for x in range(n_clusters)]
    handles = [Line2D([0],[0],color=colors[i]) for i in range(n_clusters)]
    plt.legend(handles, legend)
    plt.show()

