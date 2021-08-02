# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from extra.pipeline import run_pipeline_GCM, run_pipeline_LCGA, run_pipeline_extended_GCM

## love dataset
total_data = np.genfromtxt("test/playground_data/benchmark6_data.csv", delimiter=",", skip_header=0)
y = total_data[:,0:4] # love scores
time = np.array([0., 2., 4., 6.])
run_pipeline_LCGA(y, time, max_degree=2, max_latent_classes=3)
