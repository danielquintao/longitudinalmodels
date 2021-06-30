# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.LCGA import LCGA
import numpy as np
from utils.lcga_plot import plot_lcga_TWO_groups, plot_lcga

total_data = np.genfromtxt("test/playground_data/lovedata.csv", delimiter=",", skip_header=1)
y = total_data[:,0:4] # love scores
time = np.array([-3,3,9,36])
degree = 1
N_classes = 2
model = LCGA(y, time, degree, N_classes)
Rs, betas, pis = model.solve()
print('R\n', Rs)
print('betas\n', betas)
print('pis', pis)

deltas_hat = model.get_clusterwise_probabilities()
plot_lcga_TWO_groups(betas, time, y, degree, deltas_hat[:,1])
preds = model.get_predictions()
plot_lcga(betas, time, y, degree, preds)
