import numpy as np
from scipy.linalg import cholesky

# benchmark 1:
# time = np.array([0,0.5,1,1.5])
# degree = 2
# R = np.array([[0.44104149,0.067189,0.39024968,0.19707457],
# [0.067189,  0.01930611,0.11374701,0.08231817],
# [0.39024968,0.11374701,1.15562417,0.67200883],
# [0.19707457,0.08231817,0.67200883,0.4710173 ]])
# D = np.array([[0.65631548,0.38231413,0.68427865],
# [0.38231413,0.63427012,0.62846982],
# [0.68427865,0.62846982,0.9453542]])

# benchmark 2:
# time = np.array([0,0.5,1,1.5])
# degree = 1
# R = np.array([[0.14564146,0.14149475,0.18150496,0.26653151],
# [0.14149475,0.42934242,0.62691399,0.74003003],
# [0.18150496,0.62691399,1.2042562,1.33535028],
# [0.26653151,0.74003003,1.33535028,2.18253052]])
# D = np.array([[0.18815823,0.29074352],
# [0.29074352,0.45080754]])

# benchmark 4
time = np.array([0,0.5,1,1.5])
degree = 2
R = np.array([[0.00465815,0.06817743,0.03524088,0.05758669],
[0.06817743,1.50686319,0.57324116,1.26548227],
[0.03524088,0.57324116,0.53116996,0.68171765],
[0.05758669,1.26548227,0.68171765,1.97197545]])
D = np.array([[0.59932416,0.45561518,0.5428226],
[0.45561518,1.23592302,0.74545321],
[0.5428226,0.74545321,0.80279142]])

T = len(time)
time = np.array(time).reshape(-1,1)
k = degree+1
Z = np.ones((T,1))
for i in range(1, k):
    Z = np.concatenate((Z,time**i), axis=1)

# check positive-definiteness of R
U = cholesky(R)
# check if diagonal elements are all positive
assert all(np.diag(U) > 0)

# check positive-definiteness of D
U = cholesky(D)
# check if diagonal elements are all positive
assert all(np.diag(U) > 0)

# check positive-definiteness of R+ZDZ^T
U = cholesky(R+Z@D@Z.T)
# check if diagonal elements are all positive
assert all(np.diag(R+Z@D@Z.T) > 0)
print(U)