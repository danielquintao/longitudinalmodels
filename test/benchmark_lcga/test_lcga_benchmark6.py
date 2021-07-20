# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from test.benchmark_lcga.build_dataframe import build_dataframe
from models.LCGA import LCGA
import numpy as np

total_data = np.genfromtxt("test/playground_data/benchmark6_data.csv", delimiter=",", skip_header=0)
y = total_data[:,0:4] # love scores
time = np.array([0., 2., 4., 6.])
degree = 1
N_classes = 2

# save a version of the data compatibe with flexmix
build_dataframe(y, 'test/benchmark_lcga/benchmark6', time, degree)


model = LCGA(y, time, degree, N_classes)
Rs, betas, pis = model.solve()

np.set_printoptions(precision=5)
print('R\n', Rs)
print('betas\n', betas)
print('pis', pis)
print("sigmas:", [np.sqrt(R[0,0]) for R in Rs])

deltas_hat = model.get_clusterwise_probabilities()
preds = model.get_predictions()

# print('posterior (deltas_hat)\n', deltas_hat)
# print('clusters (preds)\n', preds)