import numpy as np
import scipy.linalg as linalg
from utils.optimization_wrapper import gcm_FIML_minimizer
from numpy.linalg import det, inv, eigvals, matrix_rank
from scipy.optimize import approx_fprime
from utils.matrix_utils import flattened2triangular # custom file with utilities for translating matrix from/to flattened form
from utils.convert_data import convert_label

class ParentExtendedGCMSolver():
    def __init__(self, y, groups, timesteps, degree):
        """Initialize data that's required to fit GCM with FIML (discrepancy) when there are groups.

        Args:
            y (ndarray): observations : each row an individual, each column a time step
            groups (binary ndarray): membership of individuals in each group (categorical or custom one-hot)
            timesteps (array): time steps, e.g. np.array([0,1,2,3])
            degree (int): degree of the polynomial to fit
        """
        assert len(y) == len(groups)
        assert np.all(~np.isnan(y)), 'y should not contain NaNs'
        assert np.all(~np.isinf(y)), 'y should not contain np.inf\'s'
        assert np.all(~np.isnan(groups)), 'groups should not contain NaNs'
        assert np.all(~np.isinf(groups)), 'groups should not contain np.inf\'s'
        # pass groups from categorial to one-hot if necessary
        groups = groups.reshape(-1,1) if len(groups.shape) == 1 else groups
        if (groups.shape[1] == 1 and not all([g in [0,1] for g in groups])):
            print('Warning: We converted groups to another representation.')
            print('You should consider explicitly doing the same. See utils.convert_data.convert_labels')
            if not np.issubdtype(groups.dtype, np.integer):
                groups_int = groups.astype(int)
                assert np.all(groups_int == groups), 'groups entries in categorical form should belong to some np.integer dtype'
                groups = groups_int
            groups = convert_label(groups, offset=np.min(groups, axis=None))
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
        # some other statistics, per group:
        self.mus_per_group = []
        self.Ss_per_group = []
        self.Ns_per_group = []
        curr = np.zeros(self.N_groups)
        y_ = y[np.all(groups==curr, axis=1)]
        self.mus_per_group.append(np.mean(y_, axis=0))
        self.Ss_per_group.append(np.cov(y_, rowvar=False, bias=True))
        self.Ns_per_group.append(len(y_))
        for i in range(0, self.N_groups):
            curr = np.zeros(self.N_groups)
            curr[i] = 1
            y_ = y[np.all(groups==curr, axis=1)]
            self.mus_per_group.append(np.mean(y_, axis=0))
            self.Ss_per_group.append(np.cov(y_, rowvar=False, bias=True))
            self.Ns_per_group.append(len(y_))
        self.loglik = None # loglikelihood of the last call to solve()

    def discrepancy_to_loglik(self, val): # FIXME NOT YIELDING SAME RESULTS AS LAVAAN (while basic GCM does)
        val += np.log(det(self.S)) + self.T + self.N_groups + (self.T+self.N_groups)*np.log(2*np.pi)
        return -(self.N/2)*val

    def get_loglikelihood(self):
        """get log-likelihood of the already-fitted model

        Returns:
            scalar: log-likelihood
        """
        assert self.loglik is not None, "likelihood of model called before fitting"
        return self.loglik

    def pretty_beta(self, betas_opt):
        betas = [betas_opt[0:self.k].reshape(-1,1)]
        for i in range(1,self.N_groups+1): # recall that N_groups is actually the nb of group vars
            betas.append(betas[0]+betas_opt[i*self.k:(i+1)*self.k].reshape(-1,1))
        return betas

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
        (self.mu_bar-mu_hat).T @ inv_sigma_hat @ (self.mu_bar-mu_hat) - # NOTE self.mu_bar-mu_hat is 1D and ".T" does nothing, but this yields the right result :) 
        self.T - self.N_groups)
        return f

    def loglikelihood(self, theta):
        # recover beta, R, D:
        betas_per_group = self.pretty_beta(theta[0:self.p])
        R = np.eye(self.T) * (theta[self.p:self.p+self.T] ** 2)
        D_upper = flattened2triangular(theta[self.p+self.T:], self.k)
        D = D_upper.T @ D_upper
        # auxiliary variables
        Sigma_hat = R + self.Z@ D @self.Z.T
        inv_sigma_hat = inv(Sigma_hat)
        log_det_sigma_hat = np.log(det(Sigma_hat))
        # loglikelihood (in a matricial form)
        loglik = (-1/2) * ( 
        sum([Ng * np.trace(S @ inv_sigma_hat) for Ng,S in zip(self.Ns_per_group,self.Ss_per_group)])
        + self.N * (log_det_sigma_hat + self.T*np.log(2*np.pi))
        )
        for Ng, mu_bar, beta in zip(self.Ns_per_group,self.mus_per_group,betas_per_group):
            mu = mu_bar.reshape(-1,1) # fix shape
            loglik -= 1/2 * Ng * (mu - self.Z @ beta).T @ inv_sigma_hat @  (mu - self.Z @ beta)
        return loglik[0][0]

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T + self.T*self.N_groups - self.p
        df_vars_covars = self.T*(self.T+1)//2 - self.T - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def get_n_params(self):
        """get the number of non-superfluous parameters

        Returns:
            scalar: number of non-superfluous parameters
        """
        return self.p + self.k*(self.k+1)//2 + self.T # resp betas, D, R

    def check_identifiability(self):
        # we'll check identifiability as if there was ONE group (so we'll use k instead of p below)
        dim_in = self.T+self.k*(self.k+1)//2
        dim_out = self.T*(self.T+1)//2
        if dim_in > dim_out:
            return False
        def f(theta):
            R = np.eye(self.T) * theta[:self.T]
            D_upper = flattened2triangular(theta[self.T:], self.k)
            D = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
            Sigma_hat = R + self.Z @ (D @ self.Z.T) 
            return np.tril(Sigma_hat).flatten()
        jac = np.zeros((dim_out, dim_in))
        theta = np.zeros(dim_in)
        for i in range(dim_out):
            g = lambda theta : f(theta)[i]
            jac[i,:] = approx_fprime(theta, g, 1E-8)
        return matrix_rank(jac) == dim_in

    def solve(self, verbose=True, force_solver=False, betas_pretty=False):
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
            print("intercept, slope and whatever higher degree params: {}".format(self.pretty_beta(beta_opt) if betas_pretty else beta_opt))
            print("R", R_opt)
            print("D", D_opt)

        if any(linalg.eigvalsh(R_opt) < 0):
            text = "WARNING: the residuals cov matrix (R) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(R_opt)))
            print(text)
        if any(linalg.eigvalsh(D_opt) < 0):
            text = "WARNING: the random effects cov matrix (D) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(D_opt)))
            print(text)

        # store log-likelihood
        self.loglik = self.loglikelihood(theta_opt) # self.discrepancy_to_loglik(self.discrepancy(theta_opt))

        if betas_pretty:
            return self.pretty_beta(beta_opt), R_opt, D_opt    
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
        (self.mu_bar-mu_hat).T @ inv_sigma_hat @ (self.mu_bar-mu_hat) - # NOTE self.mu_bar-mu_hat is 1D and ".T" does nothing, but this yields the right result :)
        self.T - self.N_groups)
        return f

    def loglikelihood(self, theta):
        # recover beta, R, D:
        betas_per_group = self.pretty_beta(theta[0:self.p])
        R_sigma = theta[self.p] ** 2 # to positivate
        R = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta[self.p+1:], self.k)
        D = D_upper.T @ D_upper
        # auxiliary variables
        Sigma_hat = R + self.Z@ D @self.Z.T
        inv_sigma_hat = inv(Sigma_hat)
        log_det_sigma_hat = np.log(det(Sigma_hat))
        # loglikelihood (in a matricial form)
        loglik = (-1/2) * ( 
        sum([Ng * np.trace(S @ inv_sigma_hat) for Ng,S in zip(self.Ns_per_group,self.Ss_per_group)])
        + self.N * (log_det_sigma_hat + self.T*np.log(2*np.pi))
        )
        for Ng, mu_bar, beta in zip(self.Ns_per_group,self.mus_per_group,betas_per_group):
            mu = mu_bar.reshape(-1,1) # fix shape
            loglik -= 1/2 * Ng * (mu - self.Z @ beta).T @ inv_sigma_hat @  (mu - self.Z @ beta)
        return loglik[0][0]

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T + self.T*self.N_groups - self.p
        df_vars_covars = self.T*(self.T+1)//2 - 1 - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def get_n_params(self):
        """get the number of non-superfluous parameters

        Returns:
            scalar: number of non-superfluous parameters
        """
        return self.p + self.k*(self.k+1)//2 + 1 # resp betas, D, R

    def check_identifiability(self):
        # we'll check identifiability as if there was ONE group (so we'll use k instead of p below)
        return (self.T > self.k and matrix_rank(self.Z) == self.k)

    def solve(self, verbose=True, force_solver=False, betas_pretty=False):
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

        identifiable = self.check_identifiability()
        if not force_solver:
            assert identifiable, "Identifiability problem"
        elif not identifiable:
            print('WARNING: Identifiability problem')

        if verbose:
            self.degrees_of_freedom(verbose=True) # prints d.f.

        # minimize discrepancy
        self.not_pos_def_warning_flag = False
        # f = lambda theta : -self.loglikelihood(theta)
        theta_opt, success = gcm_FIML_minimizer(self.discrepancy, [self.p,1,(self.k,self.k)], verbose=verbose)
        assert success, "WARNING: optimization did not converge\n"

        # recover optimal beta, R, D
        beta_opt = theta_opt[0:self.p]
        R_sigma = theta_opt[self.p] ** 2 # to positivate
        R_opt = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta_opt[self.p+1:], self.k)
        D_opt = D_upper.T @ D_upper
        if verbose:
            print("intercept, slope and whatever higher degree params: {}".format(self.pretty_beta(beta_opt) if betas_pretty else beta_opt))
            print("R", R_opt)
            print("D", D_opt)

        if any(linalg.eigvalsh(R_opt) < 0):
            text = "WARNING: the residuals cov matrix (R) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(R_opt)))
            print(text)
        if any(linalg.eigvalsh(D_opt) < 0):
            text = "WARNING: the random effects cov matrix (D) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(D_opt)))
            print(text)

        # store log-likelihood
        self.loglik = self.loglikelihood(theta_opt) # self.discrepancy_to_loglik(self.discrepancy(theta_opt))

        if betas_pretty:
            return self.pretty_beta(beta_opt), R_opt, D_opt    
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
        (self.mu_bar-mu_hat).T @ inv_sigma_hat @ (self.mu_bar-mu_hat) - # NOTE self.mu_bar-mu_hat is 1D and ".T" does nothing, but this yields the right result :) 
        self.T - self.N_groups)
        if f < 0:
            return 0 # the discrepancy func should be always non-negative; lavaan does this as well
        return f

    def loglikelihood(self, theta):
        # recover beta, R, D:
        betas_per_group = self.pretty_beta(theta[0:self.p])
        R = np.eye(self.T) * theta[self.p:self.p+self.T]
        D_upper = flattened2triangular(theta[self.p+self.T:], self.k)
        D = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
        # auxiliary variables
        Sigma_hat = R + self.Z@ D @self.Z.T
        inv_sigma_hat = inv(Sigma_hat)
        log_det_sigma_hat = np.log(det(Sigma_hat))
        # loglikelihood (in a matricial form)
        loglik = (-1/2) * ( 
        sum([Ng * np.trace(S @ inv_sigma_hat) for Ng,S in zip(self.Ns_per_group,self.Ss_per_group)])
        + self.N * (log_det_sigma_hat + self.T*np.log(2*np.pi))
        )
        for Ng, mu_bar, beta in zip(self.Ns_per_group,self.mus_per_group,betas_per_group):
            mu = mu_bar.reshape(-1,1) # fix shape
            loglik -= 1/2 * Ng * (mu - self.Z @ beta).T @ inv_sigma_hat @  (mu - self.Z @ beta)
        return loglik[0][0]

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T + self.T*self.N_groups - self.p
        df_vars_covars = self.T*(self.T+1)//2 - self.T - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def get_n_params(self):
        """get the number of non-superfluous parameters

        Returns:
            scalar: number of non-superfluous parameters
        """
        return self.p + self.k*(self.k+1)//2 + self.T # resp betas, D, R

    def check_identifiability(self):
        # we'll check identifiability as if there was ONE group (so we'll use k instead of p below)
        dim_in = self.T+self.k*(self.k+1)//2
        dim_out = self.T*(self.T+1)//2
        if dim_in > dim_out:
            return False
        def f(theta):
            R = np.eye(self.T) * theta[:self.T]
            D_upper = flattened2triangular(theta[self.T:], self.k)
            D = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
            Sigma_hat = R + self.Z @ (D @ self.Z.T) 
            return np.tril(Sigma_hat).flatten()
        jac = np.zeros((dim_out, dim_in))
        theta = np.zeros(dim_in)
        for i in range(dim_out):
            g = lambda theta : f(theta)[i]
            jac[i,:] = approx_fprime(theta, g, 1E-8)
        return matrix_rank(jac) == dim_in

    def solve(self, verbose=True, force_solver=False, betas_pretty=False):
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
            print("intercept, slope and whatever higher degree params: {}".format(self.pretty_beta(beta_opt) if betas_pretty else beta_opt))
            print("R", R_opt)
            print("D", D_opt)

        if any(linalg.eigvalsh(R_opt) < 0):
            text = "WARNING: the residuals cov matrix (R) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(R_opt)))
            print(text)
        if any(linalg.eigvalsh(D_opt) < 0):
            text = "WARNING: the random effects cov matrix (D) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(D_opt)))
            print(text)

        # store log-likelihood
        self.loglik = self.loglikelihood(theta_opt) # self.discrepancy_to_loglik(self.discrepancy(theta_opt))

        if betas_pretty:
            return self.pretty_beta(beta_opt), R_opt, D_opt    
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
        (self.mu_bar-mu_hat).T @ inv_sigma_hat @ (self.mu_bar-mu_hat) - # NOTE self.mu_bar-mu_hat is 1D and ".T" does nothing, but this yields the right result :) 
        self.T - self.N_groups)
        if f < 0:
            return 0 # the discrepancy func should be always non-negative; lavaan does this as well
        return f

    def loglikelihood(self, theta):
        # recover beta, R, D:
        betas_per_group = self.pretty_beta(theta[0:self.p])
        R_sigma = theta[self.p]
        R = R_sigma * np.eye(self.T)
        D_upper = flattened2triangular(theta[self.p+1:], self.k)
        D = D_upper + D_upper.T - np.eye(self.k)*np.diag(D_upper)
        # auxiliary variables
        Sigma_hat = R + self.Z@ D @self.Z.T
        inv_sigma_hat = inv(Sigma_hat)
        log_det_sigma_hat = np.log(det(Sigma_hat))
        # loglikelihood (in a matricial form)
        loglik = (-1/2) * ( 
        sum([Ng * np.trace(S @ inv_sigma_hat) for Ng,S in zip(self.Ns_per_group,self.Ss_per_group)])
        + self.N * (log_det_sigma_hat + self.T*np.log(2*np.pi))
        )
        for Ng, mu_bar, beta in zip(self.Ns_per_group,self.mus_per_group,betas_per_group):
            mu = mu_bar.reshape(-1,1) # fix shape
            loglik -= 1/2 * Ng * (mu - self.Z @ beta).T @ inv_sigma_hat @  (mu - self.Z @ beta)
        return loglik[0][0]

    def degrees_of_freedom(self, verbose=False):
        df_beta = self.T + self.T*self.N_groups - self.p
        df_vars_covars = self.T*(self.T+1)//2 - 1 - self.k*(self.k+1)//2
        if verbose:
            print("Total df: {} ({} for beta, {} for (co)variances)".format(df_beta+df_vars_covars, df_beta, df_vars_covars))
        return df_beta, df_vars_covars

    def get_n_params(self):
        """get the number of non-superfluous parameters

        Returns:
            scalar: number of non-superfluous parameters
        """
        return self.p + self.k*(self.k+1)//2 + 1 # resp betas, D, R

    def check_identifiability(self):
        # we'll check identifiability as if there was ONE group (so we'll use k instead of p below)
        return (self.T > self.k and matrix_rank(self.Z) == self.k)

    def solve(self, verbose=True, force_solver=False, betas_pretty=False):
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
            print("intercept, slope and whatever higher degree params: {}".format(self.pretty_beta(beta_opt) if betas_pretty else beta_opt))
            print("R", R_opt)
            print("D", D_opt)

        if any(linalg.eigvalsh(R_opt) < 0):
            text = "WARNING: the residuals cov matrix (R) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(R_opt)))
            print(text)
        if any(linalg.eigvalsh(D_opt) < 0):
            text = "WARNING: the random effects cov matrix (D) is not definite positive (the smallest eigenvalue is {})".format(min(linalg.eigvalsh(D_opt)))
            print(text)

        # store log-likelihood
        self.loglik = self.loglikelihood(theta_opt) # self.discrepancy_to_loglik(self.discrepancy(theta_opt))

        if betas_pretty:
            return self.pretty_beta(beta_opt), R_opt, D_opt    
        return beta_opt, R_opt, D_opt
