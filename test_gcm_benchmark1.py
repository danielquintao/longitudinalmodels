from GCM import GCMSolver
from GCM_cvxpy_failed import GCMSolver_CVXPY
from GCM_extended import ExtendedGCMSolver
import pandas as pd
import numpy as np
from gcm_plot import plot, extended_plot

total_data = np.genfromtxt("playground_data/benchmark1_data.csv", delimiter=",", skip_header=0)
# print(total_data)
y = total_data[:,0:4] # measures in time steps
time = np.array([0., 0.5, 1., 1.5]) # cf. benchmark1_ground_truth.txt

degree = 2 # cf. benchmark1_ground_truth.txt

print("=========== GCM Solver (with Scipy.optimization) ===================")

gcm = GCMSolver(y, time, degree)
beta_opt, R_opt, D_opt = gcm.solve()

plot(beta_opt, time, y, degree)

sigma = R_opt + gcm.Z @ D_opt @ gcm.Z.T
print("y cov matrix:\n{}".format(sigma))

print("==== Extended GCM Solver (known groups) (with Scipy.optimization) ====")

# We'll now test the GCM solver with known groups (predictors of fixed slope)
# In benchmark1, the last column indicates to which of the
# 2 available groups the individual belongs, and the two last columns encode it
# with 1 binary variable: (0,) = group 1, (1,) = group 2
groups = total_data[:,-1:]
# print(groups)

egcm = ExtendedGCMSolver(y, groups, time, degree)
ebeta_opt, eR_opt, eD_opt = egcm.solve()

extended_plot(ebeta_opt, time, y, groups, [(0,),(1,)] ,degree)

esigma = eR_opt + egcm.Z @ eD_opt @ egcm.Z.T
print("y cov matrix:\n{}".format(esigma))