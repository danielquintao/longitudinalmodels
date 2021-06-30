# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.GCM import GCM
import numpy as np
from utils.gcm_plot import plot, extended_plot

total_data = np.genfromtxt("test/playground_data/benchmark3_data.csv", delimiter=",", skip_header=0)
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
#     sgcm = GCM(y, time, degree, R_struct='diagonal')
#     sbeta_opt, sR_opt, sD_opt = sgcm.solve()

#     plot(sbeta_opt, time, y, degree)

#     ssigma = sR_opt + sgcm.Z @ sD_opt @ sgcm.Z.T
#     print("Sigma:\n{}".format(ssigma))
# except AssertionError as err:
#     print(err)

print("=========== GCM Solver w/ time-indep. errors (Scipy.optimization) ===================")

try:
    tigcm = GCM(y, time, degree)
    tibeta_opt, tiR_opt, tiD_opt = tigcm.solve()

    plot(tibeta_opt, time, y, degree)

    tisigma = tiR_opt + tigcm.Z @ tiD_opt @ tigcm.Z.T
    print("Sigma:\n{}".format(tisigma))
except AssertionError as err:
    print(err)

# print("======= GCM w/ time-indep. errors lavaan-like estimator (Scipy.optimization) =====")

# try:
#     tifigcm = GCM(y, time, degree, lavaan_like=True)
#     tifibeta_opt, tifiR_opt, tifiD_opt = tifigcm.solve()

#     plot(tifibeta_opt, time, y, degree)

#     tifisigma = tifiR_opt + tifigcm.Z @ tifiD_opt @ tifigcm.Z.T
#     print("Sigma:\n{}".format(tifisigma))
# except AssertionError as err:
#     print(err)