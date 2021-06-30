# add parent folder in order to run tests
# https://docs.python-guide.org/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.LCGA import LCGA
import numpy as np
import matplotlib.pyplot as plt

total_data = np.genfromtxt("test/playground_data/benchmark1_data.csv", delimiter=",", skip_header=0)
# TRUNCATE TO 2 TIME STEPS:
y = total_data[:,0:1] # measures in time steps
time = np.array([0., 0.5]) # cf. benchmark1_ground_truth.txt
degree = 0 # cf. benchmark1_ground_truth.txt
N_classes = 2
model = LCGA(y, time, degree, N_classes)

# simulate multivariate normal data and check function that computes PDF
beta = np.random.rand(degree+1,1)
mean = (model.X @ beta).flatten()
data = np.random.multivariate_normal(mean, np.eye(len(time)), 5000)
PDFs = []
for datum in data:
    val = model.multivar_normal_PDF(datum.reshape(-1,1), np.eye(len(time)), beta)
    assert 0 <= val <= 1, "OOPS"
    PDFs.append(val)
plt.figure()
plt.scatter(data[:,0], data[:,1], s=2, c=np.array(PDFs))
plt.show()