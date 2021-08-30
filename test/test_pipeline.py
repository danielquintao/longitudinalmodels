# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from extra.pipeline import run_pipeline_GCM, run_pipeline_LCGA, run_pipeline_extended_GCM

## love dataset
total_data = np.genfromtxt("test/playground_data/lovedata.csv", delimiter=",", skip_header=1)
y = total_data[:,0:4] # love scores
time = np.array([-3,3,9,36])
## groups as categorical number
# groups = total_data[:,-3]
# run_pipeline_GCM(y, time, 2, varname='marital love', use_log=True)
# run_pipeline_extended_GCM(y, time, 2, groups, varname='marital love', use_log=True)
# ## groups as custom one-hot
# groups2 = total_data[:,-2:]
# run_pipeline_GCM(y, time, 2)
# run_pipeline_extended_GCM(y, time, 2, groups2)
# ## LCGA
run_pipeline_LCGA(y, time, max_degree=1, max_latent_classes=3, varname='marital love', use_log=True)

# ## ultimate dataset (allows harmonization)
# main_data = np.genfromtxt("harmonizer/fake_data/ultimate_dataset_main.csv", delimiter=",", skip_header=0)
# control_data = np.genfromtxt("harmonizer/fake_data/ultimate_dataset_control.csv", delimiter=",", skip_header=0)
# labels = np.genfromtxt("harmonizer/fake_data/ultimate_dataset_labels.csv", delimiter=",", skip_header=0)
# labels_1D = np.argwhere(labels)[:,1]
# time = np.array([0,1,2,3])
# degree = 1
# N_classes = 3
# run_pipeline_GCM(main_data, time, 2, control_data, labels_1D)
