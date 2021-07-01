import numpy as np
import scipy.linalg as linalg
from numpy.linalg import det, inv, eigvals, pinv, matrix_rank
import scipy.optimize as optimize
from utils.optimization_wrapper import gcm_minimizer, gcm_FIML_minimizer
from utils.matrix_utils import flattened2triangular # custom file with utilities for translating matrix from/to flattened form
from utils.gcm_plot import plot

class ParentGCMSolver():
    def __init__(self, y, timesteps, degree):
        self.mu_bar = np.mean(y, axis=0) # sample mean
        self.S = np.cov(y, rowvar=False, bias=True) # sample covariance (divided by N i.e. biased)
        self.N = len(y)
        self.p = degree+1 # we include the intercept (coefficient of order 0)
        self.k = self.p
        self.T = len(timesteps) # time points
        self.time = timesteps
        X = np.ones((self.T,1))
        for i in range(1,degree+1): # We are using time as parameter -- TODO? custom X per individual
            X = np.concatenate((X, (self.time**i).reshape(-1,1)), axis=1)
        self.X = X
        self.Z = X

#----------------------------------------------------------------------------------------------#

class DiagGCMSolver(ParentGCMSolver):
    def __init__(self, y, timesteps, degree):
        super().__init__(y, timesteps, degree)

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
        R = np.eye(self.T) * (theta[self.p:self.p+self.T] ** 2)
        D_upper = flattened2triangular(theta[self.p+self.T:], self.k)
        D = D_upper.T @ D_upper
        Sigma_hat = R + self.Z @ (D @ self.Z.T) 
        mu_hat = (self.X @ beta).flatten()
        inv_sigma_hat = inv(Sigma_hat)
        log_det_sigma_hat = np.log(det(Sigma_hat))
        f = (log_det_sigma_hat - np.log(det(self.S)) + 
        np.trace(self.S @ inv_sigma_hat) + 
        (self.mu_bar-mu_hat).T @ inv_sigma_hat @ (self.mu_bar-mu_hat) - 
        self.T)
        return f

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T - self.p
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

class TimeIndepErrorGCMSolver(ParentGCMSolver):
    def __init__(self, y, timesteps, degree):
        super().__init__(y, timesteps, degree)

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
        R_sigma = theta[self.p] ** 2 # to positivate
        R = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta[self.p+1:], self.k)
        D = D_upper.T @ D_upper
        Sigma_hat = R + self.Z @ (D @ self.Z.T) 
        mu_hat = (self.X @ beta).flatten()
        inv_sigma_hat = inv(Sigma_hat)
        log_det_sigma_hat = np.log(det(Sigma_hat))
        f = (log_det_sigma_hat - np.log(det(self.S)) + 
        np.trace(self.S @ inv_sigma_hat) + 
        (self.mu_bar-mu_hat).T @ inv_sigma_hat @ (self.mu_bar-mu_hat) - 
        self.T)
        return f

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T - self.p 
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

class DiagGCMLavaanLikeSolver(ParentGCMSolver):
    def __init__(self, y, timesteps, degree):
        super().__init__(y, timesteps, degree)

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
        R = np.eye(self.T) * theta[self.p:self.p+self.T] # differently from our likelihood, we'll allow neg vals
        D_upper = flattened2triangular(theta[self.p+self.T:], self.k)
        D = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
        Sigma_hat = R + self.Z @ (D @ self.Z.T) 
        mu_hat = (self.X @ beta).flatten()
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
        self.T)
        if f < 0:
            return 0 # the discrepancy func should be always non-negative; lavaan does this as well
        return f

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T - self.p 
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

class TimeIndepErrorGCMLavaanLikeSolver(ParentGCMSolver):
    def __init__(self, y, timesteps, degree):
        super().__init__(y, timesteps, degree)

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
        R_sigma = theta[self.p]
        R = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta[self.p+1:], self.k)
        D = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
        Sigma_hat = R + self.Z @ (D @ self.Z.T) 
        mu_hat = (self.X @ beta).flatten()
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
        self.T)
        if f < 0:
            return 0 # the discrepancy func should be always non-negative; lavaan does this as well
        return f

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T - self.p 
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


