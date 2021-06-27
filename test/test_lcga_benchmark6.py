# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LCGA import LCGA
import numpy as np
from utils.lcga_plot import plot_lcga_TWO_groups

total_data = np.genfromtxt("test/playground_data/benchmark6_data.csv", delimiter=",", skip_header=1)
y = total_data[:,0:4] # love scores
time = np.array([0., 2., 4., 6.])
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

# 0-th EM iteration: success; eval = 2646.6549086928694
# 1-th EM iteration: success; eval = 2587.3744363565297
# 2-th EM iteration: success; eval = 2562.378548822419
# 3-th EM iteration: failed...; eval = 2562.3782226905346
# 4-th EM iteration: success; eval = 2562.378222690477
# 5-th EM iteration: success; eval = 2562.3782226904905
# R
#  [array([[13.005,  0.   ,  0.   ,  0.   ],
#        [ 0.   , 13.005,  0.   ,  0.   ],
#        [ 0.   ,  0.   , 13.005,  0.   ],
#        [ 0.   ,  0.   ,  0.   , 13.005]]), array([[7.65, 0.  , 0.  , 0.  ],
#        [0.  , 7.65, 0.  , 0.  ],
#        [0.  , 0.  , 7.65, 0.  ],
#        [0.  , 0.  , 0.  , 7.65]])]
# betas
#  [array([[32.431],
#        [ 2.145]]), array([[17.514],
#        [ 0.733]])]
# pis [0.5140562248422325, 0.4859437751577676]