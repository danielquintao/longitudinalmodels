import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_loglikelihoods(vals, key1_name, key2_name):
    """plots the log-likelihood of different models, specially for LCGA
       Example: for LCGA key1 is the degree of the model and key2 is the number of latent classes K

    Args:
        vals (dict): dict in the form {degree: {K: (loglik, n_params)}}.
                     The fact that the inner-most values are TUPLES is important.
        key1 (str): name of key1
        key2 (str): name of key2
    """
    colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple',
        5:'tab:brown', 6:'tab:pink', 7:'tab:gray', 8:'tab:olive', 9:'tab:cyan'}
    plt.figure()
    x1 = sorted(vals)
    min_x = np.inf
    max_x = -np.inf
    for i,key1 in enumerate(x1):
        x2 = sorted(vals[key1])
        y2 = [vals[key1][key2][0] for key2 in x2]
        plt.plot(x2, y2, 'o-', color=colors[i])
        min_x = min(min_x,min(x2))
        max_x = max(max_x,max(x2))
    if len(x1) > 1:
        legend = [key1_name+' '+str(key1) for key1 in x1]
        handles = [Line2D([0],[0],color=colors[i]) for i in range(len(x1))]
        plt.legend(handles, legend)
    plt.xlabel(key2_name)
    plt.ylabel('log-likelihood')
    plt.xticks(np.arange(min_x, max_x+1, 1))
    plt.title('log-likelihood per model')
    plt.show()

def plot_information_criterions(vals, key1_name, key2_name, N):
    """plots the AIC and BIC of different models, specially for LCGA
       Example: for LCGA key1 is the degree of the model and key2 is the number of latent classes K

    Args:
        vals (dict): dict in the form {degree: {K: (loglik, n_params)}}.
                     The fact that the inner-most values are TUPLES is important.
        key1 (str): name of key1
        key2 (str): name of key2
        N (int): number of individuals (for BIC)
    """
    colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple',
        5:'tab:brown', 6:'tab:pink', 7:'tab:gray', 8:'tab:olive', 9:'tab:cyan'}
    log_N = np.log(N)
    plt.figure()
    x1 = sorted(vals)
    min_x = np.inf
    max_x = -np.inf
    for i,key1 in enumerate(x1):
        x2 = sorted(vals[key1])
        aic = []
        bic = []
        for key2 in x2:
            loglik, n_params = vals[key1][key2]
            aic.append(2*(n_params-loglik))
            bic.append(log_N*n_params-2*loglik)
        plt.plot(x2, aic, 'o-', color=colors[i])
        plt.plot(x2, bic, 'o--', color=colors[i])
        min_x = min(min_x,min(x2))
        max_x = max(max_x,max(x2))
    legend = ['AIC '+key1_name+' '+str(key1) for key1 in x1]
    legend.extend(['BIC '+key1_name+' '+str(key1) for key1 in x1])
    handles = [Line2D([0],[0],color=colors[i]) for i in range(len(x1))]
    handles.extend([Line2D([0],[0],linestyle='dotted',color=colors[i]) for i in range(len(x1))])
    plt.legend(handles, legend)
    plt.xlabel(key2_name)
    plt.ylabel('quality measure (lower is better)')
    plt.xticks(np.arange(min_x, max_x+1, 1))
    plt.title('Model comparison through AIC and BIC')
    plt.show()
