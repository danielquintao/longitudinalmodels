# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GCM import GCM
import numpy as np
from utils.gcm_plot import plot, extended_plot

total_data = np.genfromtxt("test/playground_data/lovedata.csv", delimiter=",", skip_header=1)
# print(total_data)
y = total_data[:,0:4] # love scores
time = np.array([-3,3,9,36])

degree = 1

# print("=========== GCM Solver (with Scipy.optimization) ===================")

# try:
#     gcm = GCMSolver(y, time, degree)
#     beta_opt, R_opt, D_opt = gcm.solve()

#     plot(beta_opt, time, y, degree)

#     sigma = R_opt + gcm.Z @ D_opt @ gcm.Z.T
#     print("Sigma matrix:\n{}".format(sigma))

#     # gcm.multisolve()
# except AssertionError as err:
#     print(err)

print("=========== GCM Solver with diagonal R (Scipy.optimization) ===================")

try:
    sgcm = GCM(y, time, degree, R_struct='diagonal')
    sbeta_opt, sR_opt, sD_opt = sgcm.solve()

    plot(sbeta_opt, time, y, degree)

    ssigma = sR_opt + sgcm.Z @ sD_opt @ sgcm.Z.T
    print("Sigma:\n{}".format(ssigma))
except AssertionError as err:
    print(err)

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

# print("======= GCM w/ diagonal R lavaan-like estimator (Scipy.optimization) =====")

# try:
#     drfigcm = GCM(y, time, degree, R_struct='diagonal', lavaan_like=True)
#     drfibeta_opt, drfiR_opt, drfiD_opt = drfigcm.solve()

#     plot(drfibeta_opt, time, y, degree)

#     drfisigma = drfiR_opt + drfigcm.Z @ drfiD_opt @ drfigcm.Z.T
#     print("Sigma:\n{}".format(drfisigma))
# except AssertionError as err:
#     print(err)

# We'll now test the GCM solver with known groups (predictors of fixed slope)
# In lovedata dataset, the before-before-last column indicates to which of the
# 3 available groups the individual belongs, and the two last columns encode it
# with binary variables: (1,0) = group 1, (0,0) = group 2, (0,1) = group 3
groups = total_data[:,-2:]
# print(groups)

# print("==== Extended GCM Solver (known groups) (with Scipy.optimization) ====")

# try:
#     egcm = ExtendedGCMSolver(y, groups, time, degree)
#     ebeta_opt, eR_opt, eD_opt = egcm.solve()

#     extended_plot(ebeta_opt, time, y, groups, [(0,0),(1,0),(0,1)] ,degree)

#     esigma = eR_opt + egcm.Z @ eD_opt @ egcm.Z.T
#     print("Sigma:\n{}".format(esigma))
# except AssertionError as err:
#     print(err)

print("==== Extended GCM Solver (known groups) w/ diag. R (Scipy.optimization) ====")

# We'll now test the GCM solver with known groups (predictors of fixed slope)
# In lovedata dataset, the before-before-last column indicates to which of the
# 3 available groups the individual belongs, and the two last columns encode it
# with binary variables: (1,0) = group 1, (0,0) = group 2, (0,1) = group 3
groups = total_data[:,-2:]
# print(groups)

try:
    esgcm = GCM(y, time, degree, R_struct='diagonal', groups=groups)
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
    tiesgcm = GCM(y, time, degree, groups=groups)
    tiesbeta_opt, tiesR_opt, tiesD_opt = tiesgcm.solve()

    extended_plot(tiesbeta_opt, time, y, groups, [(0,0),(1,0),(0,1)] ,degree)

    tiessigma = tiesR_opt + tiesgcm.Z @ tiesD_opt @ tiesgcm.Z.T
    print("Sigma:\n{}".format(tiessigma))
except AssertionError as err:
    print(err)

# print("==== Extended GCM Solver (known groups) w/ time-indep. errors lavaan-like estimator (Scipy.optimization) ====")

# groups = total_data[:,-2:]
# # print(groups)

# try:
#     tifiesgcm = GCM(y, time, degree, groups=groups, lavaan_like=True)
#     tifiesbeta_opt, tifiesR_opt, tifiesD_opt = tifiesgcm.solve()

#     extended_plot(tifiesbeta_opt, time, y, groups, [(0,0),(1,0),(0,1)], degree)

#     tifiessigma = tifiesR_opt + tifiesgcm.Z @ tifiesD_opt @ tifiesgcm.Z.T
#     print("Sigma:\n{}".format(tifiessigma))
# except AssertionError as err:
#     print(err)

# print("==== Extended GCM Solver (known groups) w/ diagonal R lavaan-like estimator (Scipy.optimization) ====")

# groups = total_data[:,-2:]
# # print(groups)

# try:
#     drfiesgcm = GCM(y, time, degree, R_struct='diagonal', groups=groups, lavaan_like=True)
#     drfiesbeta_opt, drfiesR_opt, drfiesD_opt = drfiesgcm.solve()

#     extended_plot(drfiesbeta_opt, time, y, groups, [(0,0),(1,0),(0,1)], degree)

#     drfiessigma = drfiesR_opt + drfiesgcm.Z @ drfiesD_opt @ drfiesgcm.Z.T
#     print("Sigma:\n{}".format(drfiessigma))
# except AssertionError as err:
#     print(err)