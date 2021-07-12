# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from test.benchmark_lcga.build_dataframe import build_dataframe
from models.LCGA import LCGA
import numpy as np

# The idea of this script is to run an LCGA w/ 1 class (linear regression) to cmpare with
# the results of flexmix and of our code, since we know a closed formula for that case
# that is equivalent to using either maximum likelihood or least squares
# for thi script to make sense, run benchmark5 with flexmix and with our code using 1 class

total_data = np.genfromtxt("test/playground_data/benchmark5_data.csv", delimiter=",", skip_header=1)
y = total_data[:,0:4] # love scores
time = np.array([0., 2., 4., 6.])

X = np.ones((len(time),2))
X[:,1] = time

sX = np.tile(X, (len(y),1)) # stacked X
sy = y.reshape(-1,1) # stacked y
beta_opt = np.linalg.inv(sX.T@sX)@sX.T@sy

sigma_square = sum((sy - sX@beta_opt)**2) / (len(y)*len(time))

print('beta_opt\n', beta_opt)
print('sigma^2:\n', sigma_square)
# this way, we obtained the same results as our implementation
# however, dividing the expression of sigma^2 by len(y)*len(time)-2
# instead of simply len(y)*len(time) yields the same results as flexmix
# (maybe only coincidence)