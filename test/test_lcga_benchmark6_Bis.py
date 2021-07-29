# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.LCGA import LCGA
import numpy as np
from utils.lcga_plot import plot_lcga_TWO_groups, plot_lcga

total_data = np.genfromtxt("test/playground_data/benchmark6_data.csv", delimiter=",", skip_header=0)
y = total_data[:,0:4] # love scores
time = np.array([0., 2., 4., 6.])
degree = 1
N_classes = 3
model = LCGA(y, time, degree, N_classes, R_struct='diagonal')
Rs, betas, pis = model.solve()

print('R\n', Rs)
print('betas\n', betas)
print('pis', pis)

preds = model.get_predictions()
plot_lcga(betas, time, y, degree, preds)
