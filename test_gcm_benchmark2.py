from GCM import GCMSolver, SimplifiedGCMSolver
from GCM_cvxpy_failed import GCMSolver_CVXPY
from GCM_extended import ExtendedGCMSolver
import pandas as pd
import numpy as np
from gcm_plot import plot, extended_plot

total_data = np.genfromtxt("playground_data/benchmark2_data.csv", delimiter=",", skip_header=0)
# print(total_data)
y = total_data[:,0:4] # measures in time steps
time = np.array([0., 0.5, 1., 1.5]) # cf. benchmark2_ground_truth.txt

degree = 1 # cf. benchmark2_ground_truth.txt

print("=========== GCM Solver (with Scipy.optimization) ===================")

try:
    gcm = GCMSolver(y, time, degree)
    beta_opt, R_opt, D_opt = gcm.solve()

    plot(beta_opt, time, y, degree)

    sigma = R_opt + gcm.Z @ D_opt @ gcm.Z.T
    print("Sigma:\n{}".format(sigma))
except AssertionError as err:
    print(err)

print("=========== GCM Solver with diagonal R (Scipy.optimization) ===================")

try:
    sgcm = SimplifiedGCMSolver(y, time, degree)
    sbeta_opt, sR_opt, sD_opt = sgcm.solve()

    plot(sbeta_opt, time, y, degree)

    ssigma = sR_opt + sgcm.Z @ sD_opt @ sgcm.Z.T
    print("Sigma:\n{}".format(ssigma))
except AssertionError as err:
    print(err)