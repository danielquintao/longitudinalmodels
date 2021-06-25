# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from GCM import GCM
from statsmodels.api import MixedLM
import numpy as np
import pandas as pd


total_data = np.genfromtxt("test/playground_data/benchmark2_data.csv", delimiter=",", skip_header=0)
y = total_data[:,0:4] # measures in time steps
time = np.array([0., 0.5, 1., 1.5]) # cf. benchmark2_ground_truth.txt
degree = 1 # cf. benchmark2_ground_truth.txt

data = pd.DataFrame(y, columns=time)
data['individual'] = data.index
data = pd.melt(data, id_vars='individual', value_vars=time, var_name='time', value_name='observation')
data['time'] = pd.to_numeric(data['time'])
data['intercept'] = 1
exog_cols = ['intercept', 'time']

##############

theirs = MixedLM(
                endog=data['observation'],
                exog=data[exog_cols],
                groups=data['individual'],
                exog_re=data[exog_cols]
            )
their_fit = theirs.fit(reml=False)

beta = np.array(their_fit.fe_params)
D = np.array(their_fit.cov_re)

print(beta)
np.set_printoptions(precision=3)
print(D)

##############

try:
    gcm = GCM(y, time, degree)
    _, _, _ = gcm.solve()
except AssertionError as err:
    print(err)