from GCM import GCMSolver
from GCM_cvxpy_failed import GCMSolver_CVXPY
from GCM_extended import ExtendedGCMSolver
import pandas as pd
import numpy as np
from gcm_plot import plot, extended_plot

total_data = np.genfromtxt("playground_data/lovedata.csv", delimiter=",", skip_header=1)
# print(total_data)
y = total_data[:,0:4] # love scores
time = np.array([-3,3,9,36])

degree = 1

print("=========== GCM Solver (with Scipy.optimization) ===================")

gcm = GCMSolver(y, time, degree)
beta_opt, R_opt, D_opt = gcm.solve()

plot(beta_opt, time, y, degree)

# gcm.multisolve()

# print("=========== GCM Solver with CVXPY ===================================")

# gcm_cvxpy = GCMSolver_CVXPY(y, time, degree)
# beta_opt, R_opt, D_opt = gcm_cvxpy.solve()

# plot(beta_opt, time, y, degree)

print("==== Extended GCM Solver (known groups) (with Scipy.optimization) ====")

# We'll now test the GCM solver with known groups (predictors of fixed slope)
# In lovedata dataset, the before-before-last column indicates to which of the
# 3 available groups the individual belongs, and the two last columns encode it
# with binary variables: (1,0) = group 1, (0,0) = group 2, (0,1) = group 3
groups = total_data[:,-2:]
# print(groups)

egcm = ExtendedGCMSolver(y, groups, time, degree)
ebeta_opt, eR_opt, eD_opt = egcm.solve()

extended_plot(ebeta_opt, time, y, groups, [(0,0),(1,0),(0,1)] ,degree)