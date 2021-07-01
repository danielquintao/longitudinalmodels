import numpy as np
import scipy.linalg as linalg
from utils.optimization_wrapper import gcm_FIML_minimizer
from numpy.linalg import det, inv, eigvals, matrix_rank
from utils.matrix_utils import flattened2triangular # custom file with utilities for translating matrix from/to flattened form
from utils.convert_data import convert_label

class ParentExtendedGCMSolver():
    def __init__(self, y, groups, timesteps, degree):
        """Initialize data that's required to fit GCM with FIML (discrepancy) when there are groups.

        Args:
            y (ndarray): observations : each row an individual, each column a time step
            groups (binary ndarray): membership of individuals in each group (0 or 1)
            timesteps (array): time steps, e.g. np.array([0,1,2,3])
            degree (int): degree of the polynomial to fit
        """
        assert len(y) == len(groups)
        # pass groups from categorial to one-hot if necessary
        if ((len(groups.shape) == 1 or (len(groups.shape) == 2 and groups.shape[1] == 1))
            and not all([g in [0,1] for g in groups])):
            print('Warning: We converted groups to another representation.')
            print('You should consider explicitly doing the same. See utils.convert_data.convert_labels')
            groups = convert_label(groups, offset=min(groups))
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
        # Z is a matrix of shape (T, k)
        # [[1,1,1]]
        # [[1,2,4]]
        # [[1,3,9]]
        # [[1,4,16]]
        # for T=4, degree=2
        Z = np.ones((self.T,1))
        for i in range(1,degree+1):
            Z = np.concatenate((Z, (self.time**i).reshape(-1,1)), axis=1)
        self.Z = Z

#----------------------------------------------------------------------------------------------#

class DiagExtendedGCMSolver(ParentExtendedGCMSolver):
    def __init__(self, y, groups, timesteps, degree):
        super().__init__(y, groups, timesteps, degree)

    def discrepancy(self, theta):
        """Discrepancy funcion (Preacher et al. Latent Growth Curve Modeling), a.k.a. Full-Information ML (Bollen, Kolenikov 2008)

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
        R = np.eye(self.T) * (theta[self.p:self.p+self.T] ** 2)
        D_upper = flattened2triangular(theta[self.p+self.T:], self.k)
        D = D_upper.T @ D_upper
        # Sigma_hat, mu_hat
        Sigma_hat = np.zeros((self.T+self.N_groups,self.T+self.N_groups))
        Sigma_hat[0:self.T,0:self.T] = R + self.Z @ (D + w @ self.x_cov @ w.T ) @ self.Z.T
        Sigma_hat[self.T:,0:self.T] = self.x_cov @ w.T @ self.Z.T
        Sigma_hat[0:self.T,self.T:] = self.Z @ w @ self.x_cov
        Sigma_hat[self.T:,self.T:] = self.x_cov
        mu_hat = np.zeros(self.T+self.N_groups)
        mu_hat[0:self.T] = (self.Z @ (eta + w @ self.x_bar.reshape(-1,1))).flatten()
        mu_hat[self.T:] = self.x_bar
        inv_sigma_hat = inv(Sigma_hat)
        log_det_sigma_hat = np.log(det(Sigma_hat))
        f = (log_det_sigma_hat - np.log(det(self.S)) + 
        np.trace(self.S @ inv_sigma_hat) + 
        (self.mu_bar-mu_hat).T @ inv_sigma_hat @ (self.mu_bar-mu_hat) - 
        self.T - self.N_groups)
        return f

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T + self.T*self.N_groups - self.p
        df_vars_covars = self.T*(self.T+1)//2 - self.T - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def solve(self, verbose=True, force_solver=False):
        """estimate model

        Args:
            verbose (bool, optional): Verbose mode or not. Defaults to True.
            force_solver (bool, optional): Whether to estimate model if condition on degrees of freedom is not satisfied.
                                           Defaults to False.

        Returns:
            (1D array, 2D array, 2D array): beta_opt (fixed effects), R_opt (cov matrix of errors), D_opt (cov matrix of random effects)
                                            Note: the params of beta_opt are in the following order: fixed effects of class (0,0,..,0),
                                            then fixed effects of class (1,0,..,0), then (0,1,..,0), ..., (0,0,..,1)
        """

        if not force_solver:
            assert all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]), "Identifiability problem: you have more parameters than 'information'"
        elif not all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]):
            print('WARNING: Identifiability problem with degrees of freedom')

        if matrix_rank(self.Z) < self.k:
            print("WARNING: the random effects design matrix is not full-rank")

        # minimize discrepancy
        self.not_pos_def_warning_flag = False
        theta_opt, success = gcm_FIML_minimizer(self.discrepancy, [self.p,self.T,(self.k,self.k)], verbose=verbose)
        assert success, "WARNING: optimization did not converge"

        # recover optimal beta, R, D
        beta_opt = theta_opt[0:self.p]
        R_opt = np.eye(self.T) * (theta_opt[self.p:self.p+self.T] ** 2)
        D_upper = flattened2triangular(theta_opt[self.p+self.T:], self.k)
        D_opt = D_upper.T @ D_upper
        if verbose:
            print("intercept, slope and whatever higher degree params: {}".format(beta_opt))
            print("R", R_opt)
            print("D", D_opt)

        assert all(linalg.eigvals(R_opt) > 0), "WARNING: R is not definite-positive"
        assert all(linalg.eigvals(D_opt) > 0), "WARNING: D is not definite-positive"

        return beta_opt, R_opt, D_opt

class TimeIndepErrorExtendedGCMSolver(ParentExtendedGCMSolver):
    def __init__(self, y, groups, timesteps, degree):
        super().__init__(y, groups, timesteps, degree)

    def discrepancy(self, theta):
        """Discrepancy funcion (Preacher et al. Latent Growth Curve Modeling), a.k.a. Full-Information ML (Bollen, Kolenikov 2008)

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
        R_sigma = theta[self.p] ** 2 # to positivate
        R = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta[self.p+1:], self.k)
        D = D_upper.T @ D_upper
        # Sigma_hat, mu_hat
        Sigma_hat = np.zeros((self.T+self.N_groups,self.T+self.N_groups))
        Sigma_hat[0:self.T,0:self.T] = R + self.Z @ (D + w @ self.x_cov @ w.T ) @ self.Z.T
        Sigma_hat[self.T:,0:self.T] = self.x_cov @ w.T @ self.Z.T
        Sigma_hat[0:self.T,self.T:] = self.Z @ w @ self.x_cov
        Sigma_hat[self.T:,self.T:] = self.x_cov
        mu_hat = np.zeros(self.T+self.N_groups)
        mu_hat[0:self.T] = (self.Z @ (eta + w @ self.x_bar.reshape(-1,1))).flatten()
        mu_hat[self.T:] = self.x_bar
        inv_sigma_hat = inv(Sigma_hat)
        log_det_sigma_hat = np.log(det(Sigma_hat))
        f = (log_det_sigma_hat - np.log(det(self.S)) + 
        np.trace(self.S @ inv_sigma_hat) + 
        (self.mu_bar-mu_hat).T @ inv_sigma_hat @ (self.mu_bar-mu_hat) - 
        self.T - self.N_groups)
        return f

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T + self.T*self.N_groups - self.p
        df_vars_covars = self.T*(self.T+1)//2 - 1 - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def solve(self, verbose=True, force_solver=False):
        """estimate model

        Args:
            verbose (bool, optional): Verbose mode or not. Defaults to True.
            force_solver (bool, optional): Whether to estimate model if condition on degrees of freedom is not satisfied.
                                           Defaults to False.

        Returns:
            (1D array, 2D array, 2D array): beta_opt (fixed effects), R_opt (cov matrix of errors), D_opt (cov matrix of random effects)
                                            Note: the params of beta_opt are in the following order: fixed effects of class (0,0,..,0),
                                            then fixed effects of class (1,0,..,0), then (0,1,..,0), ..., (0,0,..,1)
        """

        if not force_solver:
            assert all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]), "Identifiability problem: you have more parameters than 'information'"
        elif not all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]):
            print('WARNING: Identifiability problem with degrees of freedom')

        if matrix_rank(self.Z) < self.k:
            print("WARNING: the random effects design matrix is not full-rank")

        # minimize discrepancy
        self.not_pos_def_warning_flag = False
        theta_opt, success = gcm_FIML_minimizer(self.discrepancy, [self.p,1,(self.k,self.k)], verbose=verbose)
        assert success, "WARNING: optimization did not converge"

        # recover optimal beta, R, D
        beta_opt = theta_opt[0:self.p]
        R_sigma = theta_opt[self.p] ** 2 # to positivate
        R_opt = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta_opt[self.p+1:], self.k)
        D_opt = D_upper.T @ D_upper
        if verbose:
            print("intercept, slope and whatever higher degree params: {}".format(beta_opt))
            print("R", R_opt)
            print("D", D_opt)

        assert all(linalg.eigvals(R_opt) > 0), "WARNING: R is not definite-positive"
        assert all(linalg.eigvals(D_opt) > 0), "WARNING: D is not definite-positive"

        return beta_opt, R_opt, D_opt

#----------------------------------------------------------------------------------------------#

class DiagExtendedGCMLavaanLikeSolver(ParentExtendedGCMSolver):
    def __init__(self, y, groups, timesteps, degree):
        super().__init__(y, groups, timesteps, degree)

    def discrepancy(self, theta):
        """Discrepancy funcion (Preacher et al. Latent Growth Curve Modeling), a.k.a. Full-Information ML (Bollen, Kolenikov 2008)

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

    def solve(self, verbose=True, force_solver=False):
        """estimate model

        Args:
            verbose (bool, optional): Verbose mode or not. Defaults to True.
            force_solver (bool, optional): Whether to estimate model if condition on degrees of freedom is not satisfied.
                                           Defaults to False.

        Returns:
            (1D array, 2D array, 2D array): beta_opt (fixed effects), R_opt (cov matrix of errors), D_opt (cov matrix of random effects)
                                            Note: the params of beta_opt are in the following order: fixed effects of class (0,0,..,0),
                                            then fixed effects of class (1,0,..,0), then (0,1,..,0), ..., (0,0,..,1)
        """

        if not force_solver:
            assert all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]), "Identifiability problem: you have more parameters than 'information'"
        elif not all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]):
            print('WARNING: Identifiability problem with degrees of freedom')

        if matrix_rank(self.Z) < self.k:
            print("WARNING: the random effects design matrix is not full-rank")

        # minimize discrepancy
        self.not_pos_def_warning_flag = False
        theta_opt, success = gcm_FIML_minimizer(self.discrepancy, [self.p,self.T,(self.k,self.k)], verbose=verbose)
        assert success, "WARNING: optimization did not converge"
        if self.not_pos_def_warning_flag:
            print("WARNING: We encountered positive-definiteness problems during optimization.")

        # recover optimal beta, R, D
        beta_opt = theta_opt[0:self.p]
        R_opt = np.eye(self.T) * theta_opt[self.p:self.p+self.T]
        D_upper = flattened2triangular(theta_opt[self.p+self.T:], self.k)
        D_opt = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
        if verbose:
            print("intercept, slope and whatever higher degree params: {}".format(beta_opt))
            print("R", R_opt)
            print("D", D_opt)

        assert all(linalg.eigvals(R_opt) > 0), "WARNING: R is not definite-positive"
        assert all(linalg.eigvals(D_opt) > 0), "WARNING: D is not definite-positive"

        return beta_opt, R_opt, D_opt

class TimeIndepErrorExtendedGCMLavaanLikeSolver(ParentExtendedGCMSolver):
    def __init__(self, y, groups, timesteps, degree):
        super().__init__(y, groups, timesteps, degree)

    def discrepancy(self, theta):
        """Discrepancy funcion (Preacher et al. Latent Growth Curve Modeling), a.k.a. Full-Information ML (Bollen, Kolenikov 2008)

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

    def solve(self, verbose=True, force_solver=False):
        """estimate model

        Args:
            verbose (bool, optional): Verbose mode or not. Defaults to True.
            force_solver (bool, optional): Whether to estimate model if condition on degrees of freedom is not satisfied.
                                           Defaults to False.

        Returns:
            (1D array, 2D array, 2D array): beta_opt (fixed effects), R_opt (cov matrix of errors), D_opt (cov matrix of random effects)
                                            Note: the params of beta_opt are in the following order: fixed effects of class (0,0,..,0),
                                            then fixed effects of class (1,0,..,0), then (0,1,..,0), ..., (0,0,..,1)
        """

        if not force_solver:
            assert all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]), "Identifiability problem: you have more parameters than 'information'"
        elif not all([x > 0 for x in self.degrees_of_freedom(verbose=verbose)]):
            print('WARNING: Identifiability problem with degrees of freedom')

        if matrix_rank(self.Z) < self.k:
            print("WARNING: the random effects design matrix is not full-rank")

        # minimize discrepancy
        self.not_pos_def_warning_flag = False
        theta_opt, success = gcm_FIML_minimizer(self.discrepancy, [self.p,1,(self.k,self.k)], verbose=verbose)
        assert success, "WARNING: optimization did not converge"
        if self.not_pos_def_warning_flag:
            print("WARNING: We encountered positive-definiteness problems during optimization.")

        # recover optimal beta, R, D
        beta_opt = theta_opt[0:self.p]
        R_sigma = theta_opt[self.p]
        R_opt = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta_opt[self.p+1:], self.k)
        D_opt = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
        if verbose:
            print("intercept, slope and whatever higher degree params: {}".format(beta_opt))
            print("R", R_opt)
            print("D", D_opt)

        assert all(linalg.eigvals(R_opt) > 0), "WARNING: R is not definite-positive"
        assert all(linalg.eigvals(D_opt) > 0), "WARNING: D is not definite-positive"

        return beta_opt, R_opt, D_opt
