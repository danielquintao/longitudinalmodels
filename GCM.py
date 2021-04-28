import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize
from matrix_utils import flattened2triangular # custom file with utilities for translating matrix from/to flattened form
from gcm_plot import plot

class ParentGCMSolver():
    def __init__(self, y, timesteps, degree):
        self.y = y
        self.N = len(y)
        self.p = degree+1 # we include the intercept (coefficient of order 0)
        self.k = self.p # we have no "fixed" predictor yet -- TODO
        self.T = len(timesteps) # time points
        self.time = timesteps
        X = np.ones((self.T,1))
        for i in range(1,degree+1): # We are using time as parameter -- TODO? custom X per individual
            X = np.concatenate((X, (self.time**i).reshape(-1,1)), axis=1)
        self.X = X
        self.Z = X # we have no "fixed" predictor yet -- TODO

class GCMSolver(ParentGCMSolver):
    def __init__(self, y, timesteps, degree):
        super().__init__(y, timesteps, degree)

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
            second_term += (self.y[i].reshape(-1,1) - self.X @ beta).T @ linalg.inv(variance) @ (self.y[i].reshape(-1,1) - self.X @ beta)
        second_term = second_term[0] * -1/2 # "[0]" because output of loop above is 1x1 2D ndarray
        return -(first_term + second_term)

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T - self.p
        df_vars_covars = self.T*(self.T+1)//2 - self.T*(self.T+1)//2 - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def solve(self, method='BFGS'):

        assert all([x > 0 for x in self.degrees_of_freedom(verbose=True)]), "Identifiability problem: you have more parameters than 'information'"

        # initial guess for the optimization
        beta_0 = np.zeros((self.p,1))
        R_upper0 = np.random.rand(int(self.T*(self.T+1)/2))
        D_upper0 = np.random.rand(int(self.k*(self.k+1)/2))
        theta_0 = np.concatenate((beta_0.flatten(), R_upper0, D_upper0))

        # maximize likelihood -- default
        if method == 'BFGS':
            optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='BFGS',
            options={'maxiter':1000, 'disp':True})
        elif method == 'TNC':
            optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='TNC',
            options={'maxfun':1000, 'disp':True})
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

    def multisolve(self):
        """Test multiple optimization methods
        With the parental love dataset (Oravecz, Muth - DOI 10.3758/s13423-017-1281-0), the
        best methods (in terms of conergence) were "Customized BFGS 2 (3-point approx for jac)"
        and "TNC 2 (3-point approx for jac)"
        """
        # initial guess for the optimization
        beta_0 = np.zeros((self.p,1))
        R_upper0 = np.random.rand(int(self.T*(self.T+1)/2))
        D_upper0 = np.random.rand(int(self.k*(self.k+1)/2))
        theta_0 = np.concatenate((beta_0.flatten(), R_upper0, D_upper0))

        # maximize likelihood -- default
        print("Default")
        optimize_res = optimize.minimize(self.minus_l, theta_0, options={'maxiter':1000, 'disp':True})
        print("Log-likelihood maximization succeeded: {}".format(optimize_res.success))
        print(optimize_res.message)

        # maximize likelihood -- Nelder-Mead
        print("Nelder-Mead")
        optimize_res = optimize.minimize(self.minus_l, theta_0, method='Nelder-Mead')
        print("Log-likelihood maximization succeeded: {}".format(optimize_res.success))
        print(optimize_res.message)

        # maximize likelihood -- customized BFGS
        print("Customized BFGS")
        optimize_res = optimize.minimize(self.minus_l, theta_0, method='BFGS',
        options={'maxiter':1000, 'eps':.00000001})
        print("Log-likelihood maximization succeeded: {}".format(optimize_res.success))
        print(optimize_res.message)

        # maximize likelihood -- customized BFGS 2
        print("Customized BFGS 2 (3-point approx for jac)")
        optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='BFGS',
        options={'maxiter':1000, 'disp':True})
        print("Log-likelihood maximization succeeded: {}".format(optimize_res.success))
        print(optimize_res.message)

        # maximize likelihood -- Truncated Newton
        print("TNC")
        optimize_res = optimize.minimize(self.minus_l, theta_0, method='TNC',
        options={'maxfun':1000, 'disp':True})
        print("Log-likelihood maximization succeeded: {}".format(optimize_res.success))
        print(optimize_res.message)

        # maximize likelihood -- Truncated Newton 2
        print("TNC 2 (3-point approx for jac)")
        optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='TNC',
        options={'maxfun':1000, 'disp':True})
        print("Log-likelihood maximization succeeded: {}".format(optimize_res.success))
        print(optimize_res.message)

class SimplifiedGCMSolver(ParentGCMSolver):
    def __init__(self, y, timesteps, degree):
        super().__init__(y, timesteps, degree)

    def minus_l(self, theta):
        """log-likelihood * (-1) (because we are in a maximization problem)

        Args:
            theta (ndarray): In the context of GCM, we expect a 1D ndarray of format
                            [beta, flattened diagonal of sqrt(R), flattened cholesky decomp. of D], 
                            where the cholesky decompositions are "upper-triangular" (scipy default)
                            and written in the flattened form a11, a12,..., a1n, a22, a23, ... 
                            (check functions flattened2triangular and triangular2flattened) 
                            Note: In order to recover the original D and R, p and T must be known globally

        Returns:
            scalar: (-1) * log-likelihood for theta under the GCM model
        """
        # recover beta, R, D:
        beta = theta[0:self.p].reshape(-1,1) # column
        R = np.eye(self.T) * (theta[self.p:self.p+self.T] ** 2)
        D_upper = flattened2triangular(theta[self.p+self.T:], self.k)
        D = D_upper.T @ D_upper
        # compute likelihood:
        variance = R + self.Z @ (D @ self.Z.T)
        # if linalg.det(variance) < 0:
        #     print("det(variance matrix) = {}".format(linalg.det(variance)))
        first_term = -(self.N/2) * np.log(linalg.det(variance))
        second_term = 0
        for i in range(self.N):
            second_term += (self.y[i].reshape(-1,1) - self.X @ beta).T @ linalg.inv(variance) @ (self.y[i].reshape(-1,1) - self.X @ beta)
        second_term = second_term[0] * -1/2 # "[0]" because output of loop above is 1x1 2D ndarray
        return -(first_term + second_term)

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T - self.p
        df_vars_covars = self.T*(self.T+1)//2 - self.T - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def solve(self, method='BFGS'):

        assert all([x > 0 for x in self.degrees_of_freedom(verbose=True)]), "Identifiability problem: you have more parameters than 'information'"

        # initial guess for the optimization
        beta_0 = np.zeros((self.p,1))
        R_diag = np.random.rand(self.T)
        D_upper0 = np.random.rand(int(self.k*(self.k+1)/2))
        theta_0 = np.concatenate((beta_0.flatten(), R_diag, D_upper0))

        # maximize likelihood -- default
        if method == 'BFGS':
            optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='BFGS',
            options={'maxiter':1000, 'disp':True})
        elif method == 'TNC':
            optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='TNC',
            options={'maxfun':1000, 'disp':True})
        else:
            print("'method' {} not recognized!".format(method))
            raise ValueError
        theta_opt = optimize_res.x
        print("Log-likelihood maximization succeeded: {}".format(optimize_res.success))
        print(optimize_res.message)

        # recover optimal beta, R, D
        beta_opt = theta_opt[0:self.p]
        R_diag = theta_opt[self.p:self.p+self.T]
        R_opt = np.eye(self.T) * (R_diag ** 2)
        D_upper = flattened2triangular(theta_opt[self.p+self.T:],self.k)
        D_opt = D_upper.T @ D_upper
        print("intercept, slope and whatever higher degree params: {}".format(beta_opt))
        print("R", R_opt)
        print("D", D_opt)

        return beta_opt, R_opt, D_opt
    
class TimeIndepErrorGCMSolver(ParentGCMSolver):
    def __init__(self, y, timesteps, degree):
        super().__init__(y, timesteps, degree)

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
        R_sigma = theta[self.p] ** 2 # to positivate
        R = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta[self.p+1:], self.k)
        D = D_upper.T @ D_upper
        # compute likelihood:
        variance = R + self.Z @ (D @ self.Z.T)
        # if linalg.det(variance) < 0:
        #     print("det(variance matrix) = {}".format(linalg.det(variance)))
        first_term = -(self.N/2) * np.log(linalg.det(variance))
        second_term = 0
        for i in range(self.N):
            second_term += (self.y[i].reshape(-1,1) - self.X @ beta).T @ linalg.inv(variance) @ (self.y[i].reshape(-1,1) - self.X @ beta)
        second_term = second_term[0] * -1/2 # "[0]" because output of loop above is 1x1 2D ndarray
        return -(first_term + second_term)

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T - self.p 
        df_vars_covars = self.T*(self.T+1)//2 - 1 - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def solve(self, method='BFGS'):

        assert all([x > 0 for x in self.degrees_of_freedom(verbose=True)]), "Identifiability problem: you have more parameters than 'information'"

        # initial guess for the optimization
        beta_0 = np.zeros((self.p,1))
        R_sigma0 = np.random.rand(1) + 0.00000001 # strictly positive
        D_upper0 = np.random.rand(int(self.k*(self.k+1)/2))
        theta_0 = np.concatenate((beta_0.flatten(), R_sigma0, D_upper0))

        # maximize likelihood -- default
        if method == 'BFGS':
            optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='BFGS',
            options={'maxiter':1000, 'disp':True})
        elif method == 'TNC':
            optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='TNC',
            options={'maxfun':1000, 'disp':True})
        else:
            print("'method' {} not recognized!".format(method))
            raise ValueError
        theta_opt = optimize_res.x
        print("Log-likelihood maximization succeeded: {}".format(optimize_res.success))
        print(optimize_res.message)

        # recover optimal beta, R, D
        beta_opt = theta_opt[0:self.p]
        R_sigma = theta_opt[self.p] ** 2 # to positivate
        R_opt = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta_opt[self.p+1:],self.k)
        D_opt = D_upper.T @ D_upper
        print("intercept, slope and whatever higher degree params: {}".format(beta_opt))
        print("R", R_opt)
        print("D", D_opt)

        return beta_opt, R_opt, D_opt

class UnconstrainedGCMSolver(ParentGCMSolver):
    def __init__(self, y, timesteps, degree):
        super().__init__(y, timesteps, degree)

    def minus_l(self, theta):
        """log-likelihood * (-1) (because we are in a maximization problem)

        Args:
            theta (ndarray): In the context of GCM, we expect a 1D ndarray of format
                            [beta, R, D]
                            Note: In order to recover the original D and R, p and T must be known globally

        Returns:
            scalar: (-1) * log-likelihood for theta under the GCM model
        """
        # recover beta, R, D:
        beta = theta[0:self.p].reshape(-1,1) # column
        R_sigma = theta[self.p]
        R = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta[self.p+1:], self.k)
        D = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
        # compute likelihood:
        variance = R + self.Z @ (D @ self.Z.T)
        # if linalg.det(variance) < 0:
        #     print("det(variance matrix) = {}".format(linalg.det(variance)))
        first_term = -(self.N/2) * np.log(linalg.det(variance))
        second_term = 0
        for i in range(self.N):
            second_term += (self.y[i].reshape(-1,1) - self.X @ beta).T @ linalg.inv(variance) @ (self.y[i].reshape(-1,1) - self.X @ beta)
        second_term = second_term[0] * -1/2 # "[0]" because output of loop above is 1x1 2D ndarray
        return -(first_term + second_term)

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T - self.p 
        df_vars_covars = self.T*(self.T+1)//2 - 1 - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def solve(self, method='BFGS'):

        assert all([x > 0 for x in self.degrees_of_freedom(verbose=True)]), "Identifiability problem: you have more parameters than 'information'"

        # initial guess for the optimization
        beta_0 = np.zeros((self.p,1))
        R_sigma0 = np.random.rand(1) + 0.00000001 # strictly positive
        D_0 = np.random.rand(self.k*(self.k+1)//2)
        theta_0 = np.concatenate((beta_0.flatten(), R_sigma0, D_0))

        # maximize likelihood -- default
        if method == 'BFGS':
            optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='BFGS',
            options={'maxiter':1000, 'disp':True})
        elif method == 'TNC':
            optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='TNC',
            options={'maxfun':1000, 'disp':True})
        else:
            print("'method' {} not recognized!".format(method))
            raise ValueError
        theta_opt = optimize_res.x
        print("Log-likelihood maximization succeeded: {}".format(optimize_res.success))
        print(optimize_res.message)

        # recover optimal beta, R, D
        beta_opt = theta_opt[0:self.p]
        R_sigma = theta_opt[self.p]
        R_opt = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta_opt[self.p+1:], self.k)
        D_opt = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
        print("intercept, slope and whatever higher degree params: {}".format(beta_opt))
        print("R", R_opt)
        print("D", D_opt)

        return beta_opt, R_opt, D_opt