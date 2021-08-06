import numpy as np
import scipy.linalg as linalg
from numpy.linalg import det, inv, eigvals, pinv, matrix_rank
from scipy.optimize import approx_fprime
from utils.optimization_wrapper import gcm_minimizer, gcm_FIML_minimizer
from utils.matrix_utils import flattened2triangular # custom file with utilities for translating matrix from/to flattened form
from utils.gcm_plot import plot

class ParentGCMSolver():
    def __init__(self, y, timesteps, degree):
        assert np.all(~np.isnan(y)), 'y should not contain NaNs'
        assert np.all(~np.isinf(y)), 'y should not contain np.inf\'s'
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
        self.loglik = None # loglikelihood of the last call to solve()

    def discrepancy_to_loglik(self, val):
        val += np.log(det(self.S)) + self.T + + self.T*np.log(2*np.pi)
        return -(self.N/2)*val

    def get_loglikelihood(self):
        """get log-likelihood of the already-fitted model

        Returns:
            scalar: log-likelihood
        """
        assert self.loglik is not None, "likelihood of model called before fitting"
        return self.loglik

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

    def get_n_params(self):
        """get the number of non-superfluous parameters

        Returns:
            scalar: number of non-superfluous parameters
        """
        return self.k + self.k*(self.k+1)//2 + self.T # resp beta, D, R

    def check_identifiability(self):
        dim_in = self.T+self.p*(self.p+1)//2
        dim_out = self.T*(self.T+1)//2
        if dim_in > dim_out:
            return False
        def f(theta):
            R = np.eye(self.T) * theta[:self.T]
            D_upper = flattened2triangular(theta[self.T:], self.p)
            D = D_upper + D_upper.T - np.eye(self.p)*np.diag(D_upper)
            Sigma_hat = R + self.Z @ (D @ self.Z.T) 
            return np.tril(Sigma_hat).flatten()
        jac = np.zeros((dim_out, dim_in))
        theta = np.zeros(dim_in)
        for i in range(dim_out):
            g = lambda theta : f(theta)[i]
            jac[i,:] = approx_fprime(theta, g, 1E-8)
        return matrix_rank(jac) == dim_in

    def solve(self, verbose=True, force_solver=False):
        """estimate model

        Args:
            verbose (bool, optional): Verbose mode or not. Defaults to True.
            force_solver (bool, optional): Whether to estimate model if condition on degrees of freedom is not satisfied.
                                           Defaults to False.

        Returns:
            (1D array, 2D array, 2D array): beta_opt (fixed effects), R_opt (cov matrix of errors), D_opt (cov matrix of random effects)
        """

        identifiable = self.check_identifiability()
        if not force_solver:
            assert identifiable, "Identifiability problem"
        elif not identifiable:
            print('WARNING: Identifiability problem')

        if verbose:
            self.degrees_of_freedom(verbose=True) # prints d.f.

        # minimize discrepancy
        self.not_pos_def_warning_flag = False
        theta_opt, success = gcm_FIML_minimizer(self.discrepancy, [self.p,self.T,(self.k,self.k)], verbose=verbose)
        assert success, "WARNING: optimization did not converge\n"

        # recover optimal beta, R, D
        beta_opt = theta_opt[0:self.p]
        R_opt = np.eye(self.T) * (theta_opt[self.p:self.p+self.T] ** 2)
        D_upper = flattened2triangular(theta_opt[self.p+self.T:], self.k)
        D_opt = D_upper.T @ D_upper
        if verbose:
            print("intercept, slope and whatever higher degree params: {}".format(beta_opt))
            print("R", R_opt)
            print("D", D_opt)

        if any(linalg.eigvalsh(R_opt) < 0):
            text = "WARNING: the residuals cov matrix (R) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(R_opt)))
            if verbose:
                print(text)
            else:
                raise Warning(text)
        if any(linalg.eigvalsh(D_opt) < 0):
            text = "WARNING: the random effects cov matrix (D) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(D_opt)))
            if verbose:
                print(text)
            else:
                raise Warning(text)

        # store log-likelihood
        self.loglik = self.discrepancy_to_loglik(self.discrepancy(theta_opt))

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

    def get_n_params(self):
        """get the number of non-superfluous parameters

        Returns:
            scalar: number of non-superfluous parameters
        """
        return self.k + self.k*(self.k+1)//2 + 1 # resp beta, D, R

    def check_identifiability(self):
        return (self.T > self.p and matrix_rank(self.Z) == self.p)

    def solve(self, verbose=True, force_solver=False):
        """estimate model

        Args:
            verbose (bool, optional): Verbose mode or not. Defaults to True.
            force_solver (bool, optional): Whether to estimate model if condition on degrees of freedom is not satisfied.
                                           Defaults to False.

        Returns:
            (1D array, 2D array, 2D array): beta_opt (fixed effects), R_opt (cov matrix of errors), D_opt (cov matrix of random effects)
        """

        identifiable = self.check_identifiability()
        if not force_solver:
            assert identifiable, "Identifiability problem"
        elif not identifiable:
            print('WARNING: Identifiability problem')

        if verbose:
            self.degrees_of_freedom(verbose=True) # prints d.f.

        # minimize discrepancy
        self.not_pos_def_warning_flag = False
        theta_opt, success = gcm_FIML_minimizer(self.discrepancy, [self.p,1,(self.k,self.k)], verbose=verbose)
        assert success, "WARNING: optimization did not converge\n"

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

        if any(linalg.eigvalsh(R_opt) < 0):
            text = "WARNING: the residuals cov matrix (R) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(R_opt)))
            if verbose:
                print(text)
            else:
                raise Warning(text)
        if any(linalg.eigvalsh(D_opt) < 0):
            text = "WARNING: the random effects cov matrix (D) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(D_opt)))
            if verbose:
                print(text)
            else:
                raise Warning(text)

        # store log-likelihood
        self.loglik = self.discrepancy_to_loglik(self.discrepancy(theta_opt))

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

    def get_n_params(self):
        """get the number of non-superfluous parameters

        Returns:
            scalar: number of non-superfluous parameters
        """
        return self.k + self.k*(self.k+1)//2 + self.T # resp beta, D, R

    def check_identifiability(self):
        dim_in = self.T+self.p*(self.p+1)//2
        dim_out = self.T*(self.T+1)//2
        if dim_in > dim_out:
            return False
        def f(theta):
            R = np.eye(self.T) * theta[:self.T]
            D_upper = flattened2triangular(theta[self.T:], self.p)
            D = D_upper + D_upper.T - np.eye(self.p)*np.diag(D_upper)
            Sigma_hat = R + self.Z @ (D @ self.Z.T) 
            return np.tril(Sigma_hat).flatten()
        jac = np.zeros((dim_out, dim_in))
        theta = np.zeros(dim_in)
        for i in range(dim_out):
            g = lambda theta : f(theta)[i]
            jac[i,:] = approx_fprime(theta, g, 1E-8)
        return matrix_rank(jac) == dim_in

    def solve(self, verbose=True, force_solver=False):
        """estimate model

        Args:
            verbose (bool, optional): Verbose mode or not. Defaults to True.
            force_solver (bool, optional): Whether to estimate model if condition on degrees of freedom is not satisfied.
                                           Defaults to False.

        Returns:
            (1D array, 2D array, 2D array): beta_opt (fixed effects), R_opt (cov matrix of errors), D_opt (cov matrix of random effects)
        """

        identifiable = self.check_identifiability()
        if not force_solver:
            assert identifiable, "Identifiability problem"
        elif not identifiable:
            print('WARNING: Identifiability problem')

        if verbose:
            self.degrees_of_freedom(verbose=True) # prints d.f.

        # minimize discrepancy
        self.not_pos_def_warning_flag = False
        theta_opt, success = gcm_FIML_minimizer(self.discrepancy, [self.p,self.T,(self.k,self.k)], verbose=verbose)
        assert success, "WARNING: optimization did not converge\n"
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

        if any(linalg.eigvalsh(R_opt) < 0):
            text = "WARNING: the residuals cov matrix (R) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(R_opt)))
            if verbose:
                print(text)
            else:
                raise Warning(text)
        if any(linalg.eigvalsh(D_opt) < 0):
            text = "WARNING: the random effects cov matrix (D) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(D_opt)))
            if verbose:
                print(text)
            else:
                raise Warning(text)

        # store log-likelihood
        self.loglik = self.discrepancy_to_loglik(self.discrepancy(theta_opt))

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

    def get_n_params(self):
        """get the number of non-superfluous parameters

        Returns:
            scalar: number of non-superfluous parameters
        """
        return self.k + self.k*(self.k+1)//2 + 1 # resp beta, D, R

    def check_identifiability(self):
        return (self.T > self.p and matrix_rank(self.Z) == self.p)

    def solve(self, verbose=True, force_solver=False):
        """estimate model

        Args:
            verbose (bool, optional): Verbose mode or not. Defaults to True.
            force_solver (bool, optional): Whether to estimate model if condition on degrees of freedom is not satisfied.
                                           Defaults to False.

        Returns:
            (1D array, 2D array, 2D array): beta_opt (fixed effects), R_opt (cov matrix of errors), D_opt (cov matrix of random effects)
        """

        identifiable = self.check_identifiability()
        if not force_solver:
            assert identifiable, "Identifiability problem"
        elif not identifiable:
            print('WARNING: Identifiability problem')

        if verbose:
            self.degrees_of_freedom(verbose=True) # prints d.f.

        # minimize discrepancy
        self.not_pos_def_warning_flag = False
        theta_opt, success = gcm_FIML_minimizer(self.discrepancy, [self.p,1,(self.k,self.k)], verbose=verbose)
        assert success, "WARNING: optimization did not converge\n"
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

        if any(linalg.eigvalsh(R_opt) < 0):
            text = "WARNING: the residuals cov matrix (R) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(R_opt)))
            if verbose:
                print(text)
            else:
                raise Warning(text)
        if any(linalg.eigvalsh(D_opt) < 0):
            text = "WARNING: the random effects cov matrix (D) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(D_opt)))
            if verbose:
                print(text)
            else:
                raise Warning(text)

        # store log-likelihood
        self.loglik = self.discrepancy_to_loglik(self.discrepancy(theta_opt))

        return beta_opt, R_opt, D_opt


