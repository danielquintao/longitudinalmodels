# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from models.GCM import GCM
from statsmodels.api import MixedLM
import numpy as np
import pandas as pd


total_data = np.genfromtxt("test/playground_data/benchmark3_data.csv", delimiter=",", skip_header=0)
y = total_data[:,0:5] # measures in time steps
time = np.array([0., 0.5, 1., 1.5, 2.]) # cf. benchmark3_ground_truth.txt
degree = 2 # cf. benchmark3_ground_truth.txt

data = pd.DataFrame(y, columns=time)
data['individual'] = data.index
data = pd.melt(data, id_vars='individual', value_vars=time, var_name='time', value_name='observation')
data['time'] = pd.to_numeric(data['time'])
data['time2'] = data['time'] ** 2
data['intercept'] = 1
exog_cols = ['intercept', 'time', 'time2']

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
    gcm = GCM(y, time, degree, lavaan_like=True)
    _, _, _ = gcm.solve()
except AssertionError as err:
    print(err)
