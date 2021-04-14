from GCM import GCMSolver
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

gcm = GCMSolver(y, time, degree)
beta_opt, R_opt, D_opt = gcm.solve()

plot(beta_opt, time, y, degree)

sigma = R_opt + gcm.Z @ D_opt @ gcm.Z.T
print("y cov matrix:\n{}".format(sigma))
