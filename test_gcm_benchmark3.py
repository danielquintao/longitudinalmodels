from GCM import GCMSolver, SimplifiedGCMSolver, TimeIndepErrorGCMSolver, UnconstrainedGCMSolver
from GCM import TimeIndepErrorGCMFullInformationSolver
from GCM_cvxpy_failed import GCMSolver_CVXPY
from GCM_extended import ExtendedGCMSolver
import pandas as pd
import numpy as np
from gcm_plot import plot, extended_plot

total_data = np.genfromtxt("playground_data/benchmark3_data.csv", delimiter=",", skip_header=0)
# print(total_data)
y = total_data[:,0:5] # measures in time steps
time = np.array([0., 0.5, 1., 1.5, 2.]) # cf. benchmark3_ground_truth.txt

degree = 2 # cf. benchmark3_ground_truth.txt

# print("=========== GCM Solver (with Scipy.optimization) ===================")

# try:
#     gcm = GCMSolver(y, time, degree)
#     beta_opt, R_opt, D_opt = gcm.solve()

#     plot(beta_opt, time, y, degree)

#     sigma = R_opt + gcm.Z @ D_opt @ gcm.Z.T
#     print("Sigma:\n{}".format(sigma))
# except AssertionError as err:
#     print(err)

# print("=========== GCM Solver with diagonal R (Scipy.optimization) ===================")

# try:
#     sgcm = SimplifiedGCMSolver(y, time, degree)
#     sbeta_opt, sR_opt, sD_opt = sgcm.solve()

#     plot(sbeta_opt, time, y, degree)

#     ssigma = sR_opt + sgcm.Z @ sD_opt @ sgcm.Z.T
#     print("Sigma:\n{}".format(ssigma))
# except AssertionError as err:
#     print(err)

print("=========== GCM Solver w/ time-indep. errors (Scipy.optimization) ===================")

try:
    tigcm = TimeIndepErrorGCMSolver(y, time, degree)
    tibeta_opt, tiR_opt, tiD_opt = tigcm.solve()

    plot(tibeta_opt, time, y, degree)

    tisigma = tiR_opt + tigcm.Z @ tiD_opt @ tigcm.Z.T
    print("Sigma:\n{}".format(tisigma))
except AssertionError as err:
    print(err)

print("======= GCM w/ time-indep. errors Full-Information estimator (Scipy.optimization) =====")

try:
    tifigcm = TimeIndepErrorGCMFullInformationSolver(y, time, degree)
    tifibeta_opt, tifiR_opt, tifiD_opt = tifigcm.solve()

    plot(tifibeta_opt, time, y, degree)

    tifisigma = tifiR_opt + tifigcm.Z @ tifiD_opt @ tifigcm.Z.T
    print("Sigma:\n{}".format(tifisigma))
except AssertionError as err:
    print(err)