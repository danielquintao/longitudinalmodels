import numpy as np
from gcm_plot import plot
import cvxpy as cp

class GCMSolver_CVXPY():
    def __init__(self, y, timesteps, degree):
        self.y = y
        self.N = len(y)
        self.p = degree+1 # we include the intercept (coefficient of order 0)
        self.k = self.p # simple model with no "fixed" predictor
        self.T = len(timesteps) # time points
        self.time = timesteps
        X = np.ones((self.T,1))
        for i in range(1,degree+1): # We are using time as parameter -- TODO? custom X per individual
            X = np.concatenate((X, (self.time**i).reshape(-1,1)), axis=1)
        self.X = X
        self.Z = X # simple model with no "fixed" predictor

    def solve(self):
        beta = cp.Variable(shape=(self.p))
        R = cp.Variable(shape=(self.T,self.T), PSD=True)
        D = cp.Variable(shape=(self.k, self.k), PSD=True)
        first_term = - (self.N/2) * cp.log_det(R+self.Z @ D @ self.Z.T)
        second_term = 0
        for i in range(self.N):
            second_term += -(1/2) * cp.matrix_frac(self.y[i] - self.X @ beta, R+self.Z @ D @ self.Z.T)
        obj = first_term + second_term
        # debug
        print("DEBUG -- expr.1 is quasiconcave? ", first_term.is_quasiconcave())
        print("DEBUG -- expr.2 is quasiconcave? ", second_term.is_quasiconcave())
        prob = cp.Problem(cp.Maximize(obj))
        print("Is a Disciplined Convex Problem: ", prob.is_dcp())
        prob.solve(verbose=False)
        print("Status: ", prob.status)
        print("beta: ", beta.value)
        print("R: ", R.value)
        print("D: ", D.value)
        return beta.value, R.value, D.value