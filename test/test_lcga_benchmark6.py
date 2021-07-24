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
N_classes = 2
model = LCGA(y, time, degree, N_classes, R_struct='diagonal')
Rs, betas, pis = model.solve()

# # test
# Rs2, betas2, pis2 = model.solve(step_M_per_class=False)
# print([R - R2 for R,R2 in zip(Rs,Rs2)])
# print([beta - beta2 for beta,beta2 in zip(betas,betas2)])
# print([beta - beta2 for beta,beta2 in zip(betas,betas2)])


print('R\n', Rs)
print('betas\n', betas)
print('pis', pis)

deltas_hat = model.get_clusterwise_probabilities()
plot_lcga_TWO_groups(betas, time, y, degree, deltas_hat[:,1])
preds = model.get_predictions()
plot_lcga(betas, time, y, degree, preds)

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