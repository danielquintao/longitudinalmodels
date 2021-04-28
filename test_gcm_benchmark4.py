from GCM import GCMSolver
from GCM_cvxpy_failed import GCMSolver_CVXPY
from GCM_extended import ExtendedGCMSolver, ExtendedAndSimplifiedGCMSolver, TimeIndepErrorExtendedGCMSolver
import pandas as pd
import numpy as np
from gcm_plot import plot, extended_plot

### Benchmark4 has basically the same idea as benchmark1, but with much more data ###

total_data = np.genfromtxt("playground_data/benchmark4_data.csv", delimiter=",", skip_header=0)
# print(total_data)
y = total_data[:,0:4] # measures in time steps
time = np.array([0., 0.5, 1., 1.5]) # cf. benchmark4_ground_truth.txt

degree = 2 # cf. benchmark4_ground_truth.txt

print("==== Extended GCM Solver (known groups) (with Scipy.optimization) ====")

# We'll now test the GCM solver with known groups (predictors of fixed slope)
# In benchmark1, the last column indicates to which of the
# 2 available groups the individual belongs, and the two last columns encode it
# with 1 binary variable: (0,) = group 1, (1,) = group 2
groups = total_data[:,-1:]
# print(groups)

try:
    egcm = ExtendedGCMSolver(y, groups, time, degree)
    ebeta_opt, eR_opt, eD_opt = egcm.solve()

    extended_plot(ebeta_opt, time, y, groups, [(0,),(1,)] ,degree)

    esigma = eR_opt + egcm.Z @ eD_opt @ egcm.Z.T
    print("Sigma:\n{}".format(esigma))
except AssertionError as err:
    print(err)

print("==== Extended GCM Solver (known groups) w/ diag. R (Scipy.optimization) ====")

groups = total_data[:,-1:]
# print(groups)

try:
    esgcm = ExtendedAndSimplifiedGCMSolver(y, groups, time, degree)
    esbeta_opt, esR_opt, esD_opt = esgcm.solve()

    extended_plot(esbeta_opt, time, y, groups, [(0,),(1,)] ,degree)

    essigma = esR_opt + esgcm.Z @ esD_opt @ esgcm.Z.T
    print("Sigma:\n{}".format(essigma))
except AssertionError as err:
    print(err)

print("==== Extended GCM Solver (known groups) w/ time-indep. errors (Scipy.optimization) ====")

groups = total_data[:,-1:]
# print(groups)

try:
    tiesgcm = TimeIndepErrorExtendedGCMSolver(y, groups, time, degree)
    tiesbeta_opt, tiesR_opt, tiesD_opt = tiesgcm.solve()

    extended_plot(tiesbeta_opt, time, y, groups, [(0,),(1,)] ,degree)

    tiessigma = tiesR_opt + tiesgcm.Z @ tiesD_opt @ tiesgcm.Z.T
    print("Sigma:\n{}".format(tiessigma))
except AssertionError as err:
    print(err)