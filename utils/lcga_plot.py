import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_lcga_TWO_groups(betas, time, data, degree, probs_cluster_1):
    """plots dataset with two clusters and color corresponding to the subject's prob of belonging to cluster.

    Args:
        betas (pair (list or tuple) of (degree+1,1) numpy arrays): fixed effects of each latent class
        time (numpy array of length T): [time points (supposed to be the same for all individuals)]
        data (2D numpy array of shape (N,T)): [time-observations for the N individuals]
        degree (int): degree of the polynomial
        probs_cluster_0 (1D array): prob of each yi to belong to cluster 1 (u.e. the 2nd cluster)
    """
    assert len(betas) == 2
    assert all([beta.shape == (degree+1,1) for beta in betas])
    N,T = data.shape
    assert T == len(time)
    assert probs_cluster_1.shape in [(N,),(N,1)]
    if probs_cluster_1.shape == (N,1):
        probs_cluster_1 = probs_cluster_1.flatten() 
    plt.figure()
    # plot individual, observed curves
    for i in range(N):
        # do not plot people that belong to groups that are not present in groups2plot:
        # if tuple(groups[i]) not in groups2plot:
        #     continue
        color = plt.cm.winter(256*probs_cluster_1[i])
        plt.plot(time, data[i], color=color, linestyle='dotted', linewidth=1)
    # plot population-level curves
    interval = np.linspace(time[0],time[-1], 100)
    for _, beta in enumerate(betas):
        curve = np.zeros(100)
        for i in range(degree+1):
            curve += beta[i] * interval**i
        plt.plot(interval, curve, color=plt.cm.winter(256*_), linewidth=5)
    # legends
    plt.show()

def plot_lcga(betas, time, data, degree, clusters_pred, title=None, varname=None):
    """plots dataset with points belonging to clusters as predicted by LCGA.

    Args:
        betas (list of numpy arrays): fixed effects of each latent class
        time (numpy array of length T): [time points (supposed to be the same for all individuals)]
        data (2D numpy array of shape (N,T)): [time-observations for the N individuals]
        degree (int): degree of the polynomial
        clusters_pred (1D array of type int): most likely cluster of each individual
        title (str, optional): title for the plot. No title if None. Defaults to None.
        varname (str, optional): Name of the variable y for the plot. 'y' if None. Defaults to None.
    """
    # we'll take advantage of the similarity between our problem and to plot a GCM estimation with classes
    # and have a very similar function, differing on the data format
    N,T = data.shape
    assert T == len(time)
    assert clusters_pred.shape in [(N,),(N,1)]
    if clusters_pred.shape == (N,1):
        clusters_pred = clusters_pred.flatten() 
    if not np.issubdtype(clusters_pred.dtype, np.integer):
        clusterspred_int = clusters_pred.astype(int)
        assert np.all(clusterspred_int == clusters_pred), 'clusters_pred entries in categorical form should belong to some np.integer dtype'
        clusters_pred = clusterspred_int
    n_clusters = max(clusters_pred)+1
    assert len(betas) == n_clusters
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
    interval = np.linspace(time[0],time[-1], 100)
    for counter in range(n_clusters):
        curve = np.zeros(100)
        coeffs = np.copy(betas[counter])
        for i in range(degree+1):
            curve += coeffs[i] * interval**i
        plt.plot(interval, curve, color=colors[counter], linewidth=5)
    # legends
    legend = ['group '+str(x) for x in range(n_clusters)]
    handles = [Line2D([0],[0],color=colors[i]) for i in range(n_clusters)]
    plt.legend(handles, legend)
    plt.xlabel("time steps")
    varname = 'y' if varname is None else varname
    plt.ylabel(varname)
    if title:
        plt.title(title)
    plt.show()

