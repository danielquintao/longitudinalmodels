import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize
from optimization_wrapper import gcm_minimizer, gcm_FIML_minimizer
from numpy.linalg import det, inv, eigvals, pinv
from matrix_utils import flattened2triangular # custom file with utilities for translating matrix from/to flattened form
from gcm_plot import plot

class ParentExtendedGCMSolver():
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
        # Finally, ESTIMATE min rank of X among all N individuals. It may be useful for identifiability
        self.rank_X = min(self.T, self.k + max(np.sum(groups!=0, axis=1)))

class ExtendedGCMSolver(ParentExtendedGCMSolver):
    def __init__(self, y, groups, timesteps, degree):
        super().__init__(y, groups, timesteps, degree)

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

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T + self.T*self.N_groups - self.p
        df_vars_covars = self.T*(self.T+1)//2 - self.T*(self.T+1)//2 - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def solve(self, method='BFGS', verbose=True, force_solver=False):

        if not force_solver:
            assert all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]), "Identifiability problem: you have more parameters than 'information'"
        elif not all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]):
            print('WARNING: Identifiability problem with degrees of freedom')

        # # initial guess for the optimization
        # beta_0 = np.zeros((self.p,1))
        # R_upper0 = np.random.rand(int(self.T*(self.T+1)/2))
        # D_upper0 = np.random.rand(int(self.k*(self.k+1)/2))
        # theta_0 = np.concatenate((beta_0.flatten(), R_upper0, D_upper0))

        # # maximize likelihood -- default
        # if method == 'BFGS':
        #     optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='BFGS',
        #     options={'maxiter':1000, 'disp':verbose})
        # elif method == 'TNC':
        #     optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='TNC',
        #     options={'maxfun':1000, 'disp':verbose})
        # else:
        #     print("'method' {} not recognized!".format(method))
        #     raise ValueError
        # theta_opt = optimize_res.x
        # if not verbose: # if verbose, optimization status already printed by Scipy's optimize.minimize
        #     print(optimize_res.message)

        theta_opt = gcm_minimizer(self.minus_l, [self.p,(self.T,self.T),(self.k,self.k)], verbose=verbose)

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

class ExtendedAndSimplifiedGCMSolver(ParentExtendedGCMSolver):
    def __init__(self, y, groups, timesteps, degree):
        super().__init__(y, groups, timesteps, degree)

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
            second_term += (self.y[i].reshape(-1,1) - self.X[i] @ beta).T @ linalg.inv(variance) @ (self.y[i].reshape(-1,1) - self.X[i] @ beta)
        second_term = second_term[0] * -1/2 # "[0]" because output of loop above is 1x1 2D ndarray
        return -(first_term + second_term)

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T + self.T*self.N_groups - self.p
        df_vars_covars = self.T*(self.T+1)//2 - self.T - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def solve(self, method='BFGS', verbose=True, force_solver=False):

        if not force_solver:
            assert all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]), "Identifiability problem: you have more parameters than 'information'"
        elif not all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]):
            print('WARNING: Identifiability problem with degrees of freedom')

        # # initial guess for the optimization
        # beta_0 = np.zeros((self.p,1))
        # R_diag = np.random.rand(self.T)
        # D_upper0 = np.random.rand(int(self.k*(self.k+1)/2))
        # theta_0 = np.concatenate((beta_0.flatten(), R_diag, D_upper0))

        # # maximize likelihood -- default
        # if method == 'BFGS':
        #     optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='BFGS',
        #     options={'maxiter':1000, 'disp':verbose})
        # elif method == 'TNC':
        #     optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='TNC',
        #     options={'maxfun':1000, 'disp':verbose})
        # else:
        #     print("'method' {} not recognized!".format(method))
        #     raise ValueError
        # theta_opt = optimize_res.x
        # if not verbose: # if verbose, optimization status already printed by Scipy's optimize.minimize
        #     print(optimize_res.message)

        theta_opt = gcm_minimizer(self.minus_l, [self.p,self.T,(self.k,self.k)], verbose=verbose)

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

class TimeIndepErrorExtendedGCMSolver(ParentExtendedGCMSolver):
    def __init__(self, y, groups, timesteps, degree):
        super().__init__(y, groups, timesteps, degree)

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
            second_term += (self.y[i].reshape(-1,1) - self.X[i] @ beta).T @ linalg.inv(variance) @ (self.y[i].reshape(-1,1) - self.X[i] @ beta)
        second_term = second_term[0] * -1/2 # "[0]" because output of loop above is 1x1 2D ndarray
        return -(first_term + second_term)

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T + self.T*self.N_groups - self.p
        df_vars_covars = self.T*(self.T+1)//2 - 1 - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def solve(self, method='BFGS', verbose=True, force_solver=False):

        if not force_solver:
            assert all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]), "Identifiability problem: you have more parameters than 'information'"
        elif not all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]):
            print('WARNING: Identifiability problem with degrees of freedom')

        # # initial guess for the optimization
        # beta_0 = np.zeros((self.p,1))
        # R_sigma0 = np.random.rand(1) + 0.00000001 # strictly positive
        # D_upper0 = np.random.rand(int(self.k*(self.k+1)/2))
        # theta_0 = np.concatenate((beta_0.flatten(), R_sigma0, D_upper0))

        # # maximize likelihood -- default
        # if method == 'BFGS':
        #     optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='BFGS',
        #     options={'maxiter':1000, 'disp':verbose})
        # elif method == 'TNC':
        #     optimize_res = optimize.minimize(self.minus_l, theta_0, jac='3-point', method='TNC',
        #     options={'maxfun':1000, 'disp':verbose})
        # else:
        #     print("'method' {} not recognized!".format(method))
        #     raise ValueError
        # theta_opt = optimize_res.x
        # if not verbose: # if verbose, optimization status already printed by Scipy's optimize.minimize
        #     print(optimize_res.message)

        theta_opt = gcm_minimizer(self.minus_l, [self.p,1,(self.k,self.k)], verbose=verbose)

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

#----------------------------------------------------------------------------------------------#

class ParentExtendedGCMFullInformationSolver():
    def __init__(self, y, groups, timesteps, degree):
        """Initialize data that's required to fit GCM with FIML (discrepancy) when there are groups.

        Args:
            y (ndarray): observations : each row an individual, each column a time step
            groups (binary ndarray): membership of individuals in each group (0 or 1)
            timesteps (array): time steps, e.g. np.array([0,1,2,3])
            degree (int): degree of the polynomial to fit
        """
        assert len(y) == len(groups)
        self.mu_bar = np.mean(np.concatenate((y,groups),axis=1), axis=0) # sample mean
        self.S = np.cov(np.concatenate((y,groups),axis=1), rowvar=False, bias=True) # sample covariance (divided by N i.e. biased)
        self.x_bar = np.mean(groups, axis=0) # mean of binary vars encoding group membership
        self.x_cov = np.cov(groups, rowvar=False, bias=True)
        self.x_cov = np.array([[self.x_cov]]) if self.x_cov.shape == () else self.x_cov # we want 2D-array
        self.N = len(y)
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
        Z = np.ones((self.T,1))
        for i in range(1,degree+1):
            Z = np.concatenate((Z, (self.time**i).reshape(-1,1)), axis=1)
        self.Z = Z

class DiagExtendedGCMFullInformationSolver(ParentExtendedGCMFullInformationSolver):
    def __init__(self, y, groups, timesteps, degree):
        super().__init__(y, groups, timesteps, degree)

    def discrepancy(self, theta):
        """Discrepancy funcion (Preacher chap.1), a.k.a. Full-Information ML (Bollen, Kolenikov)
           We extended it to groups by checking how it happens in SEM (we based on lavaan)

        Args:
            theta (ndarray): In the context of GCM, we expect a 1D ndarray of format
                            [beta, R, D]
                            Note: In order to recover the original D and R, p and T must be known globally

        Returns:
            scalar: discrepancy function (FIML) for theta under the GCM model; The lower, the better
        """
        # recover beta, R, D:
        beta = theta[0:self.p].reshape(-1,1) # column
        eta = beta[0:self.k] # main coefficients
        w = beta[self.k:].reshape(self.N_groups, self.k).T # Attention: reshape(N_groups,k).T is NOT the same as reshape(k,N_groups)
        R = np.eye(self.T) * theta[self.p:self.p+self.T]
        D_upper = flattened2triangular(theta[self.p+self.T:], self.k)
        D = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
        # Sigma_hat, mu_hat
        Sigma_hat = np.zeros((self.T+self.N_groups,self.T+self.N_groups))
        Sigma_hat[0:self.T,0:self.T] = R + self.Z @ (D + w @ self.x_cov @ w.T ) @ self.Z.T
        Sigma_hat[self.T:,0:self.T] = self.x_cov @ w.T @ self.Z.T
        Sigma_hat[0:self.T,self.T:] = self.Z @ w @ self.x_cov
        Sigma_hat[self.T:,self.T:] = self.x_cov
        mu_hat = np.zeros(self.T+self.N_groups)
        mu_hat[0:self.T] = (self.Z @ (eta + w @ self.x_bar.reshape(-1,1))).flatten()
        mu_hat[self.T:] = self.x_bar
        # check if Sigma_hat is positive-definite. We do it like lavaan
        ev = eigvals(Sigma_hat)
        if any(ev < np.sqrt(np.finfo(Sigma_hat.dtype).eps)) or sum(ev) == 0:
            self.not_pos_def_warning_flag = True
            return np.inf
        else:
            inv_sigma_hat = inv(Sigma_hat)
            log_det_sigma_hat = np.log(det(Sigma_hat))
        f = (log_det_sigma_hat - np.log(det(self.S)) + 
        np.trace(self.S @ inv_sigma_hat) + 
        (self.mu_bar-mu_hat).T @ inv_sigma_hat @ (self.mu_bar-mu_hat) - 
        self.T - self.N_groups)
        if f < 0:
            return 0 # the discrepancy func should be always non-negative; lavaan does this as well
        return f

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T + self.T*self.N_groups - self.p
        df_vars_covars = self.T*(self.T+1)//2 - self.T - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def solve(self, method='BFGS', verbose=True, force_solver=False):

        if not force_solver:
            assert all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]), "Identifiability problem: you have more parameters than 'information'"
        elif not all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]):
            print('WARNING: Identifiability problem with degrees of freedom')

        # # initial guess for the optimization
        # beta_0 = np.random.rand(self.p).reshape(-1,1) # np.zeros((self.p,1))
        # R_sigma0 = np.random.rand(T)
        # temp = np.random.rand(self.k, self.k) + 0.0001*np.ones((self.k, self.k))
        # D_0 = (temp.T @ temp)[np.triu_indices(self.k)].flatten() # make D_0 definite-positive
        # theta_0 = np.concatenate((beta_0.flatten(), R_sigma0, D_0))

        # minimize discrepancy
        self.not_pos_def_warning_flag = False
        # if method == 'BFGS':
        #     optimize_res = optimize.minimize(self.discrepancy, theta_0, jac='3-point', method='BFGS',
        #     options={'maxiter':1000, 'disp':verbose})
        # elif method == 'TNC':
        #     optimize_res = optimize.minimize(self.discrepancy, theta_0, jac='3-point', method='TNC',
        #     options={'maxfun':1000, 'disp':verbose})
        # else:
        #     print("'method' {} not recognized!".format(method))
        #     raise ValueError
        # theta_opt = optimize_res.x
        # if not verbose: # if verbose, optimization status already printed by Scipy's optimize.minimize
        #     print(optimize_res.message)
        theta_opt = gcm_FIML_minimizer(self.discrepancy, [self.p,self.T,(self.k,self.k)], verbose=verbose)
        if self.not_pos_def_warning_flag:
            print("WARNING: We encountered positive-definiteness problems during optimization.")

        # recover optimal beta, R, D
        beta_opt = theta_opt[0:self.p]
        R_opt = np.eye(self.T) * theta_opt[self.p:self.p+self.T]
        D_upper = flattened2triangular(theta_opt[self.p+self.T:], self.k)
        D_opt = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
        print("intercept, slope and whatever higher degree params: {}".format(beta_opt))
        print("R", R_opt)
        print("D", D_opt)

        assert all(linalg.eigvals(R_opt) > 0), "WARNING: R is not definite-positive"
        assert all(linalg.eigvals(D_opt) > 0), "WARNING: D is not definite-positive"

        return beta_opt, R_opt, D_opt

class TimeIndepErrorExtendedGCMFullInformationSolver(ParentExtendedGCMFullInformationSolver):
    def __init__(self, y, groups, timesteps, degree):
        super().__init__(y, groups, timesteps, degree)

    def discrepancy(self, theta):
        """Discrepancy funcion (Preacher chap.1), a.k.a. Full-Information ML (Bollen, Kolenikov)
           We extended it to groups by checking how it happens in SEM (we based on lavaan)

        Args:
            theta (ndarray): In the context of GCM, we expect a 1D ndarray of format
                            [beta, R, D]
                            Note: In order to recover the original D and R, p and T must be known globally

        Returns:
            scalar: discrepancy function (FIML) for theta under the GCM model; The lower, the better
        """
        # recover beta, R, D:
        beta = theta[0:self.p].reshape(-1,1) # column
        eta = beta[0:self.k] # main coefficients
        w = beta[self.k:].reshape(self.N_groups, self.k).T # Attention: reshape(N_groups,k).T is NOT the same as reshape(k,N_groups)
        R_sigma = theta[self.p]
        R = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta[self.p+1:], self.k)
        D = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
        # Sigma_hat, mu_hat
        Sigma_hat = np.zeros((self.T+self.N_groups,self.T+self.N_groups))
        Sigma_hat[0:self.T,0:self.T] = R + self.Z @ (D + w @ self.x_cov @ w.T ) @ self.Z.T
        Sigma_hat[self.T:,0:self.T] = self.x_cov @ w.T @ self.Z.T
        Sigma_hat[0:self.T,self.T:] = self.Z @ w @ self.x_cov
        Sigma_hat[self.T:,self.T:] = self.x_cov
        mu_hat = np.zeros(self.T+self.N_groups)
        mu_hat[0:self.T] = (self.Z @ (eta + w @ self.x_bar.reshape(-1,1))).flatten()
        mu_hat[self.T:] = self.x_bar
        # check if Sigma_hat is positive-definite. We do it like lavaan
        ev = eigvals(Sigma_hat)
        if any(ev < np.sqrt(np.finfo(Sigma_hat.dtype).eps)) or sum(ev) == 0:
            self.not_pos_def_warning_flag = True
            return np.inf
        else:
            inv_sigma_hat = inv(Sigma_hat)
            log_det_sigma_hat = np.log(det(Sigma_hat))
        f = (log_det_sigma_hat - np.log(det(self.S)) + 
        np.trace(self.S @ inv_sigma_hat) + 
        (self.mu_bar-mu_hat).T @ inv_sigma_hat @ (self.mu_bar-mu_hat) - 
        self.T - self.N_groups)
        if f < 0:
            return 0 # the discrepancy func should be always non-negative; lavaan does this as well
        return f

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T + self.T*self.N_groups - self.p
        df_vars_covars = self.T*(self.T+1)//2 - 1 - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def solve(self, method='BFGS', verbose=True, force_solver=False):

        if not force_solver:
            assert all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]), "Identifiability problem: you have more parameters than 'information'"
        elif not all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]):
            print('WARNING: Identifiability problem with degrees of freedom')

        # # initial guess for the optimization
        # beta_0 = np.random.rand(self.p).reshape(-1,1) # np.zeros((self.p,1))
        # R_sigma0 = np.random.rand(1) + 0.0001 # strictly positive
        # temp = np.random.rand(self.k, self.k) + 0.0001*np.ones((self.k, self.k))
        # D_0 = (temp.T @ temp)[np.triu_indices(self.k)].flatten() # make D_0 definite-positive
        # theta_0 = np.concatenate((beta_0.flatten(), R_sigma0, D_0))

        # minimize discrepancy
        self.not_pos_def_warning_flag = False
        # if method == 'BFGS':
        #     optimize_res = optimize.minimize(self.discrepancy, theta_0, jac='3-point', method='BFGS',
        #     options={'maxiter':1000, 'disp':verbose})
        # elif method == 'TNC':
        #     optimize_res = optimize.minimize(self.discrepancy, theta_0, jac='3-point', method='TNC',
        #     options={'maxfun':1000, 'disp':verbose})
        # else:
        #     print("'method' {} not recognized!".format(method))
        #     raise ValueError
        # theta_opt = optimize_res.x
        # if not verbose: # if verbose, optimization status already printed by Scipy's optimize.minimize
        #     print(optimize_res.message)
        theta_opt = gcm_FIML_minimizer(self.discrepancy, [self.p,1,(self.k,self.k)], verbose=verbose)
        if self.not_pos_def_warning_flag:
            print("WARNING: We encountered positive-definiteness problems during optimization.")

        # recover optimal beta, R, D
        beta_opt = theta_opt[0:self.p]
        R_sigma = theta_opt[self.p]
        R_opt = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta_opt[self.p+1:], self.k)
        D_opt = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
        print("intercept, slope and whatever higher degree params: {}".format(beta_opt))
        print("R", R_opt)
        print("D", D_opt)

        assert all(linalg.eigvals(R_opt) > 0), "WARNING: R is not definite-positive"
        assert all(linalg.eigvals(D_opt) > 0), "WARNING: D is not definite-positive"

        return beta_opt, R_opt, D_opt
