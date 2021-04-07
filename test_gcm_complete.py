import numpy as np
import pandas as pd
import scipy.linalg as linalg
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from matrix_utils import flattened2triangular # custom file with utilities for translating matrix from/to flattened form
from gcm_plot import plot

total_data = np.genfromtxt("playground_data/lovedata.csv", delimiter=",", skip_header=1)
print(total_data)
y = total_data[:,0:4] # love scores

# The information above is interesting but I will start simple
# positivity_group = total_data[:,4]
# grouping = total_data[:,5:7]
# X1 = total_data[:,5] # equivalent to np.array([x==1 for x in positivity_group])
# X2 = total_data[:,6] # equivalent to np.array([x==3 for x in positivity_group])

# params and hyperparams in the notation of appendix of van der Net et. al
# (An overview of mixture modelling for latent evolutions in longitudinal data...)
N = len(total_data)
p = 2 # we include the intercept (coefficient of order 0)
k = p # we have no "fixed" predictor
T = 4 # time points
time = np.array([-3,3,9,36])
X = np.concatenate((np.ones((T,1)), time.reshape(-1,1)), axis=1) # we'll use the time as parameter
Z = X

# log-likelihood for GCM function WITH SAME X FOR EVERY INDIVIDUAL
def l(theta):
    """log-likelihood

    Args:
        theta (ndarray): In the context of GCM, we expect a 1D ndarray of format
                         [beta, flattened cholesky decomp. of R, flattened cholesky decomp. of D], 
                         where the cholesky decompositions are "upper-triangular" (scipy default)
                         and written in the flattened form a11, a12,..., a1n, a22, a23, ... 
                         (check functions flattened2triangular and triangular2flattened) 
                         Note: In order to recover the original D and R, p and T must be known globally

    Returns:
        scalar: log-likelihood for theta under the GCM model
    """
    # recover beta, R, D:
    beta = theta[0:p].reshape(-1,1) # column
    R_upper = flattened2triangular(theta[p:p+int(T*(T+1)/2)], T)
    R = R_upper.T @ R_upper
    D_upper = flattened2triangular(theta[p+int(T*(T+1)/2):], k)
    D = D_upper.T @ D_upper
    # compute likelihood:
    variance = R + Z @ (D @ Z.T)
    # if linalg.det(variance) < 0:
    #     print("det(variance matrix) = {}".format(linalg.det(variance)))
    first_term = -(N/2) * np.log(linalg.det(variance))
    second_term = 0
    for i in range(N):
        second_term += (y[i].reshape(-1,1) - X @ beta).T @ linalg.inv(variance) @ (y[i].reshape(-1,1) - X @ beta)
    second_term = second_term[0] * -1/2 # "[0]" because output of loop above is 1x1 2D ndarray
    return first_term + second_term

# We want to maximize l, c.a.d. to minimize -l
def minus_l(theta):
    return -l(theta)

# Constraints
# P.S,: Being symmetric and positive-definite is a sufficient and necessary
# condition for a matrix to be a covariance matrix. Since we used a Cholesky
# decomposition, this is already warranted :)

# initial guess for the optimization
beta_0 = np.zeros((p,1))
R_upper0 = np.random.rand(int(T*(T+1)/2))
D_upper0 = np.random.rand(int(k*(k+1)/2))
theta_0 = np.concatenate((beta_0.flatten(), R_upper0, D_upper0))

# maximize likelihood -- default
optimize_res = optimize.minimize(minus_l, theta_0, options={'maxiter':200})
theta_opt = optimize_res.x
print("Log-likelihood maximization succeeded: {}".format(optimize_res.success))
print(optimize_res.message)

# recover optimal beta, R, D
beta_opt = theta_opt[0:p].reshape(-1,1)
R_upper = flattened2triangular(theta_opt[p:p+int(T*(T+1)/2)],T)
R_opt = R_upper.T @ R_upper
D_upper = flattened2triangular(theta_opt[p+int(T*(T+1)/2):],k)
D_opt = D_upper.T @ D_upper
print("intercept and slope: ({}, {})".format(beta_opt[0], beta_opt[1]))

# visualize linear trend
plot(beta_opt.flatten(), time, y, 1)

