import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize
from matrix_utils import flattened2triangular # custom file with utilities for translating matrix from/to flattened form
from gcm_plot import plot

class ExtendedGCMSolver():
    def __init__(self, y, groups, timesteps, degree):
        assert len(y) == len(groups)
        self.y = y
        self.N = len(y)
        self.groups = groups
        self.N_groups = groups.shape[1] # actually this is not the nb of groups stricto sensu
        self.p = (degree+1) * (1+self.N_groups)
        self.k = degree+1
        self.T = len(timesteps) # time points
        self.time = timesteps
        # X will be a tensor of shape (N, T, p)
        # We'll first build a matrix of shape (T, k), e.g.:
        # [[1,1,1]]
        # [[1,2,4]]
        # [[1,3,9]]
        # [[1,4,16]]
        # for T=4, degree=2
        X = np.ones((self.T,1))
        for i in range(1,degree+1):
            X = np.concatenate((X, (self.time**i).reshape(-1,1)), axis=1)
        self.Z = X
        # Now, we'll replicate it N times
        X2 = np.ones((self.N,self.T,self.k))
        for i in range(self.N):
            X2[i] = X
        # And then we extend the tensor in 3rd axis to bring fixed predictors to the scene, e.g.:
        # [[1,1,1]]       [[1,1, 1,xi,1*xi, 1*xi]]
        # [[1,2,4]]  -->  [[1,2, 4,xi,2*xi, 4*xi]]
        # [[1,3,9]]       [[1,3, 9,xi,3*xi, 9*xi]]
        # [[1,4,16]] x N  [[1,4,16,xi,4*xi,16*xi]] i=0,..,N-1
        # for T=4, degree=2, and one binary variable xi={0,1} (individual i belongs to group 1 or 2)
        X = np.tile(X2, (1,1,1+self.N_groups))
        for i in range(1,self.N_groups+1):
            X[:,:, i*self.k : (i+1)*self.k] *= np.tile(groups[:,i-1].reshape(self.N,1,1), (1,self.T,self.k))
        self.X = X

    def minus_l(self, theta):
        """log-likelihood * (-1) (because we are in a maximization problem)

        Args:
            theta (ndarray): In the context of GCM, we expect a 1D ndarray of format
                            [beta, flattened cholesky decomp. of R, flattened cholesky decomp. of D], 
                            where the cholesky decompositions are "upper-triangular" (scipy default)
                            and written in the flattened form a11, a12,..., a1n, a22, a23, ... 
                            (check functions flattened2triangular and triangular2flattened) 
                            Note: In order to recover the original D and R, p and T must be known globally

        Returns:
            scalar: (-1) * log-likelihood for theta under the GCM model
        """
        # recover beta, R, D:
        beta = theta[0:self.p].reshape(-1,1) # column
        R_upper = flattened2triangular(theta[self.p:self.p+int(self.T*(self.T+1)/2)], self.T)
        R = R_upper.T @ R_upper
        D_upper = flattened2triangular(theta[self.p+int(self.T*(self.T+1)/2):], self.k)
        D = D_upper.T @ D_upper
        # compute likelihood:
        variance = R + self.Z @ (D @ self.Z.T)
        # if linalg.det(variance) < 0:
        #     print("det(variance matrix) = {}".format(linalg.det(variance)))
        first_term = -(self.N/2) * np.log(linalg.det(variance))
        second_term = 0
        for i in range(self.N):
            second_term += (self.y[i].reshape(-1,1) - self.X[i] @ beta).T @ linalg.inv(variance) @ (self.y[i].reshape(-1,1) - self.X[i] @ beta)
        second_term = second_term[0] * -1/2 # "[0]" because output of loop above is 1x1 2D ndarray
        return -(first_term + second_term)

    def solve(self, method='BFGS'):
        # initial guess for the optimization
        beta_0 = np.zeros((self.p,1))
        R_upper0 = np.random.rand(int(self.T*(self.T+1)/2))
        D_upper0 = np.random.rand(int(self.k*(self.k+1)/2))
        theta_0 = np.concatenate((beta_0.flatten(), R_upper0, D_upper0))

        # maximize likelihood -- default
        if method == 'BFGS':
            optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='BFGS',
            options={'maxiter':1000})
        elif method == 'TNC':
            optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='TNC',
            options={'maxfun':1000})
        else:
            print("'method' {} not recognized!".format(method))
            raise ValueError
        theta_opt = optimize_res.x
        print("Log-likelihood maximization succeeded: {}".format(optimize_res.success))
        print(optimize_res.message)

        # recover optimal beta, R, D
        beta_opt = theta_opt[0:self.p]
        R_upper = flattened2triangular(theta_opt[self.p:self.p+int(self.T*(self.T+1)/2)],self.T)
        R_opt = R_upper.T @ R_upper
        D_upper = flattened2triangular(theta_opt[self.p+int(self.T*(self.T+1)/2):],self.k)
        D_opt = D_upper.T @ D_upper
        print("intercept, slope and whatever higher degree params: {}".format(beta_opt))
        print("R", R_opt)
        print("D", D_opt)

        return beta_opt, R_opt, D_opt