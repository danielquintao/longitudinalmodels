from GCM import GCMSolver, SimplifiedGCMSolver, TimeIndepErrorGCMSolver
from GCM_cvxpy_failed import GCMSolver_CVXPY
from GCM_extended import ExtendedGCMSolver, ExtendedAndSimplifiedGCMSolver, TimeIndepErrorExtendedGCMSolver
import pandas as pd
import numpy as np
from gcm_plot import plot, extended_plot

total_data = np.genfromtxt("playground_data/lovedata.csv", delimiter=",", skip_header=1)
# print(total_data)
y = total_data[:,0:4] # love scores
time = np.array([-3,3,9,36])

degree = 1

print("=========== GCM Solver (with Scipy.optimization) ===================")

try:
    gcm = GCMSolver(y, time, degree)
    beta_opt, R_opt, D_opt = gcm.solve()

    plot(beta_opt, time, y, degree)

    sigma = R_opt + gcm.Z @ D_opt @ gcm.Z.T
    print("Sigma matrix:\n{}".format(sigma))

    # gcm.multisolve()
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

try:
    egcm = ExtendedGCMSolver(y, groups, time, degree)
    ebeta_opt, eR_opt, eD_opt = egcm.solve()

    extended_plot(ebeta_opt, time, y, groups, [(0,0),(1,0),(0,1)] ,degree)

    esigma = eR_opt + egcm.Z @ eD_opt @ egcm.Z.T
    print("Sigma:\n{}".format(esigma))
except AssertionError as err:
    print(err)

print("==== Extended GCM Solver (known groups) w/ diag. R (Scipy.optimization) ====")

# We'll now test the GCM solver with known groups (predictors of fixed slope)
# In lovedata dataset, the before-before-last column indicates to which of the
# 3 available groups the individual belongs, and the two last columns encode it
# with binary variables: (1,0) = group 1, (0,0) = group 2, (0,1) = group 3
groups = total_data[:,-2:]
# print(groups)

try:
    esgcm = ExtendedAndSimplifiedGCMSolver(y, groups, time, degree)
    esbeta_opt, esR_opt, esD_opt = esgcm.solve()

    extended_plot(esbeta_opt, time, y, groups, [(0,0),(1,0),(0,1)] ,degree)

    essigma = esR_opt + esgcm.Z @ esD_opt @ esgcm.Z.T
    print("Sigma:\n{}".format(essigma))
except AssertionError as err:
    print(err)

print("==== Extended GCM Solver (known groups) w/ time-indep. errors (Scipy.optimization) ====")

groups = total_data[:,-2:]
# print(groups)

try:
    tiesgcm = TimeIndepErrorExtendedGCMSolver(y, groups, time, degree)
    tiesbeta_opt, tiesR_opt, tiesD_opt = tiesgcm.solve()

    extended_plot(tiesbeta_opt, time, y, groups, [(0,0),(1,0),(0,1)] ,degree)

    tiessigma = tiesR_opt + tiesgcm.Z @ tiesD_opt @ tiesgcm.Z.T
    print("Sigma:\n{}".format(tiessigma))
except AssertionError as err:
    print(err)