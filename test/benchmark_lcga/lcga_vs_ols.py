#####################################
### check LCGA by comparing with  ###
### ordinary lest squares         ###
### (should have the same result) ###
#####################################

# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from test.benchmark_lcga.build_dataframe import build_dataframe
from models.LCGA import LCGA
import numpy as np

N_classes = 1

counter = 1
def run_comparison(y, time, degree):
    global counter
    # LCGA with 1 class
    model = LCGA(y, time, degree, N_classes)
    Rs, betas, _ = model.solve(verbose=False)
    # OLS
    X = np.ones((len(time),degree+1))
    for i in range(1,degree+1):
        X[:,i] = time**i
    beta_ols = np.linalg.inv(X.T@X)@X.T@np.mean(y, axis=0).reshape(-1,1)
    residuals = y - (X@beta_ols).reshape(1,-1)
    sigma_ols = np.sqrt(np.mean(residuals * residuals, axis=None))
    # print
    print('Test {}, degree {}'.format(counter, degree))
    print('beta LCGA\n', betas[0])
    print('beta OLS \n', beta_ols)
    print("sigma LCGA", np.sqrt(Rs[0][0,0]))
    print("sigma OLS ", sigma_ols)
    print()
    counter += 1

### benchmark 5
total_data = np.genfromtxt("test/playground_data/benchmark5_data.csv", delimiter=",", skip_header=0)
y = total_data[:,0:4] # love scores
time = np.array([0., 2., 4., 6.])
run_comparison(y, time, 1)
run_comparison(y, time, 2)

### benchmark 6
total_data = np.genfromtxt("test/playground_data/benchmark6_data.csv", delimiter=",", skip_header=0)
y = total_data[:,0:4] # love scores
time = np.array([0., 2., 4., 6.])
degree = 1
run_comparison(y, time, 1)
run_comparison(y, time, 2)

### benchmark 7
total_data = np.genfromtxt("test/playground_data/benchmark7_data.csv", delimiter=",", skip_header=0)
y = total_data[:,0:4] # love scores
time = np.array([0., 0.5, 1., 1.5])
degree = 1
run_comparison(y, time, 1)
run_comparison(y, time, 2)