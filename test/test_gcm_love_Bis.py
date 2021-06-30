# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.GCM import GCM
import numpy as np
from utils.gcm_plot import extended_plot
from utils.convert_data import convert_label

total_data = np.genfromtxt("test/playground_data/lovedata.csv", delimiter=",", skip_header=1)
# print(total_data)
y = total_data[:,0:4] # love scores
time = np.array([-3,3,9,36])

degree = 1

print("==== Extended GCM Solver (known groups) w/ diag. R (Scipy.optimization) ====")

# We'll now test the GCM solver with known groups (predictors of fixed slope)
groups_categoric = total_data[:,-3].astype('int')

esgcm = GCM(y, time, degree, R_struct='diagonal', groups=groups_categoric)
esbeta_opt, esR_opt, esD_opt = esgcm.solve()

extended_plot(esbeta_opt, time, y, convert_label(groups_categoric, offset=1),
    [(0,0),(1,0),(0,1)] ,degree)

essigma = esR_opt + esgcm.Z @ esD_opt @ esgcm.Z.T
print("Sigma:\n{}".format(essigma))

print("==== Extended GCM Solver (known groups) w/ time-indep. errors (Scipy.optimization) ====")

groups_categoric = total_data[:,-3].astype('int')

try:
    tiesgcm = GCM(y, time, degree, groups=groups_categoric)
    tiesbeta_opt, tiesR_opt, tiesD_opt = tiesgcm.solve()

    extended_plot(tiesbeta_opt, time, y, convert_label(groups_categoric, offset=1),
        [(0,0),(1,0),(0,1)] ,degree)

    tiessigma = tiesR_opt + tiesgcm.Z @ tiesD_opt @ tiesgcm.Z.T
    print("Sigma:\n{}".format(tiessigma))
except AssertionError as err:
    print(err)
