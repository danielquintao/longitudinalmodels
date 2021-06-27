# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LCGA import LCGA
import numpy as np
from utils.lcga_plot import plot_lcga_TWO_groups

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
# eta = np.concatenate((betas[0],betas[1]-betas[0]), axis=0).flatten() # HACK to reuse the 'extended_plot' 
# print('eta', eta)
# extended_plot(eta, time, y, np.zeros((len(y),1)), [(0,),(1,)], 1)
def responsibility(yi):
    return pis[0]*model.multivar_normal_PDF(yi, Rs[0], betas[0]) / sum(pis[0]*model.multivar_normal_PDF(yi, Rs[0], betas[0])+pis[1]*model.multivar_normal_PDF(yi, Rs[1], betas[1]))
plot_lcga_TWO_groups(betas, time, y, degree, responsibility)