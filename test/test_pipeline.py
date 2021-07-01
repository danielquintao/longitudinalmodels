# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from extra.pipeline import run_pipeline

total_data = np.genfromtxt("test/playground_data/lovedata.csv", delimiter=",", skip_header=1)
y = total_data[:,0:4] # love scores
time = np.array([-3,3,9,36])

run_pipeline(y, time, 2)