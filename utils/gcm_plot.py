import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils.lcga_plot import plot_lcga

def plot(vector_eta, time, data, degree, title=None, varname=None):
    """plots dataset growth curves along with GCM curve

    Args:
        vector_eta (1D ndarray of length degree+1): curve coefficients
        time (list or numpy array of length T): [time points (supposed to be the same for all individuals)]
        data (2D ndarray of shape (N,T)): [time-observations for the N individuals]
        degree (int): degree of the polynomial
        title (str, optional): title for the plot. No title if None. Defaults to None.
        varname (str, optional): Name of the variable y for the plot. 'y' if None. Defaults to None.
    """
    assert vector_eta.shape == (degree+1,)
    N,T = data.shape
    assert T == len(time)
    curve = vector_eta[0]
    interval = np.linspace(time[0],time[-1], 100)
    for i in range(1, degree+1):
        curve += vector_eta[i] * (interval ** i)
    plt.figure()
    for i in range(N):
        plt.plot(time, data[i], linewidth=1)
    plt.plot(interval, curve, 'k-', linewidth=5)
    plt.xlabel("time steps")
    varname = 'y' if varname is None else varname
    plt.ylabel(varname)
    if title:
        plt.title(title)
    plt.show()

def extended_plot(vector_eta, time, data, groups, groups2plot, degree, title=None, varname=None):
    """plots dataset growth curves along with GCM curve per group.
       The number of groups in inferred from the arguments
       There are two possible data formats. We describe the first data format below.
       The 2nd data format is that of plot_lcga, and groups2plot can be anything

    Args:
        vector_eta (1D numpy array of length (M+1)*(degree+1)): The first (degree+1) vlues are the 'central' curve coefficients
                                                                and the following batches of degree+1 consecutive values are the
                                                                deviations we need to add to have the coefficients for the groups
                                                                (e.g. if there are 2 groups encoded by 0 and 1, and degree is 1, then
                                                                the first 2 values are the curve a0+a1*t for group 0, while group 1 is
                                                                represented by (a0+a2) + (a1+a3)*x)
        time (numpy array of length T): [time points (supposed to be the same for all individuals)]
        data (2D numpy array of shape (N,T)): [time-observations for the N individuals]
        groups (2D numpy array of shape (N,M)): each group is encoded by a combination of M binary values 
        groups2plot (list of tuples): combinations of our M binary variables that should be plotted as a group. For example, we may
                                      have groups2plot = [(0,0), (1,0), (0,1)] for 3 groups represented by 2 variables.
                                      There is support to plot at most 10 groups, but code can be adapted to plot more.
        degree (int): degree of the polynomial
        title (str, optional): title for the plot. No title if None. Defaults to None.
        varname (str, optional): Name of the variable y for the plot. 'y' if None. Defaults to None.
    """
    # if groups are listed as 0,1,2,..,#groups-1 or as 1,2,..,#groups and vector_eta is a list
    # of #groups fixed effects, then we will simply call the plot_lcga function whici is more adapted
    if groups.shape in [(len(groups),), (len(groups),1)] and isinstance(vector_eta, list):
        print("calling alternative plot function, more adapted for the input format...")
        if min(groups) == 1:
            print('correcting grouping variables from 1...g to 0...(g-1) for compatibility with plot_lcga')
            groups = groups-1
        plot_lcga(vector_eta, time, data, degree, groups)
        return
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
    interval = np.linspace(time[0],time[-1], 100)
    for counter, g in enumerate(groups2plot):
        curve = np.zeros(100)
        coeffs = np.copy(vector_eta[0:degree+1])
        for i, bin_var in enumerate(g,start=1):
            coeffs += bin_var * vector_eta[i*(degree+1) : (i+1)*(degree+1)]
        for i in range(degree+1):
            curve += coeffs[i] * interval**i
        plt.plot(interval, curve, color=colors[counter], linewidth=5)
    # legends
    legend = ['group '+str(x) for x in groups2plot]
    handles = [Line2D([0],[0],color=colors[i]) for i in range(len(groups2plot))]
    plt.legend(handles, legend)
    plt.xlabel("time steps")
    varname = 'y' if varname is None else varname
    plt.ylabel(varname)
    if title:
        plt.title(title)
    plt.show()
