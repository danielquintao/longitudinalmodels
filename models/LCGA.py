import numpy as np
from scipy.optimize import minimize
from utils.lcga_plot import plot_lcga_TWO_groups, plot_lcga
import random

class LCGA():
    def __init__(self, y, timesteps, degree, N_classes, R_struct="multiple_identity"):
        # TODO asserts
        self.N = len(y)
        self.k = degree+1 # we include the intercept (coefficient of order 0)
        self.N_classes = N_classes
        self.T = len(timesteps) # time points
        self.time = timesteps
        self.R_struct = R_struct
        self.y = y
        X = np.ones((self.T,1))
        for i in range(1,degree+1): # We are using time as parameter -- TODO? custom X per individual
            X = np.concatenate((X, (self.time**i).reshape(-1,1)), axis=1)
        self.X = X
        self.deltas_hat_final = None # probs of each subject belonging to each class after fit
        self.predicitons = None # most likely class per subject after fit

    def multivar_normal_PDF(self, y, R, beta):
        # fast way of inverting R (which is either multiple of identity of just diagonal)
        inv_R = np.eye(self.T) * 1/np.diag(R)
        # fast way of computing determinant of R
        det_R = np.diag(R).prod()
        return (
                np.exp(-0.5*(y-self.X @ beta).T@inv_R@(y-self.X @ beta))
                /
                ((2*np.pi)**(self.T/2) * np.sqrt(det_R))
                )

    def parse_theta(self, theta, pis_included=True):
        """parse parameter vector (for all classes)

        Args:
            theta (1D array): parameter vector in the order: Rs (covs), betas (regression coeffs),
                              and then the classes probabilities pis (if that's the case, optional)
            pis_included (bool, optional): Whether to return the pis or not. There is no problem if
                                           the argument contains the pis and pis_included is False.
                                           Defaults to True.

        Returns:
            (list, list, (list)): Rs, betas, (pis)
        """
        Rs = []
        left = 0
        for _ in range(self.N_classes):
            if self.R_struct == 'multiple_identity':
                Rs.append(np.eye(self.T) * theta[left] ** 2)
                left += 1   
            else:
                right = left + self.T
                Rs.append(np.eye(self.T) * theta[left:right] ** 2)
                left = right
        betas = []
        for _ in range(self.N_classes):
            right = left + self.k
            betas.append(theta[left:right].reshape(-1,1))
            left = right
        if pis_included:
            pis = [theta[left + _] for _ in range(self.N_classes)]
            return Rs, betas, pis
        return Rs, betas

    def parse_theta_one_class(self, theta, pi_included=True):
        """parse parameter vector for one class

        Args:
            theta (1D array): parameter vector in the order: R (cov), beta (regression coeffs),
                              and then the class probability pi (if that's the case, optional)
            pi_included (bool, optional): Whether to returnpi or not. There is no problem if
                                          the argument contains the pi and pi_included is False.
                                          Defaults to True.

        Returns:
            (list, list, (list)): R, beta, (pi)
        """
        left = 0
        if self.R_struct == 'multiple_identity':
            R = np.eye(self.T) * theta[left] ** 2
            left += 1   
        else:
            right = left + self.T
            R = np.eye(self.T) * theta[left:right] ** 2
            left = right
        right = left + self.k
        beta = theta[left:right].reshape(-1,1)
        left = right
        if pi_included:
            pi = theta[left]
            return R, beta, pi
        return R, beta

    def E_step(self, theta):
        Rs, betas, pis = self.parse_theta(theta, pis_included=True)
        # compute delta_hat's
        deltas_hat_matrix = np.zeros((self.N, self.N_classes))
        for i in range(self.N):
            for k in range(self.N_classes):
                yi = self.y[i].reshape(-1,1)
                deltas_hat_matrix[i,k] = pis[k] * self.multivar_normal_PDF(yi, Rs[k], betas[k])
            # the above expression of sqrt(delta_hat) is incomplete: we must normalize and take the square root:
            if sum(deltas_hat_matrix[i,:]) > 1E-15: # deal with numerical approx -> small vals are stored as 0
                deltas_hat_matrix[i,:] /= sum(deltas_hat_matrix[i,:]) # NOTE may cause your deltas to be 1/0s
            else:
                deltas_hat_matrix[i,:] = 1./self.N_classes # necessary HACK !!!
        # the total probabilities of a random sunject belonging to each class
        p = np.sum(deltas_hat_matrix, axis=0)
        # compute sqrt(delta_hat)'s
        sqrt_deltas_hat_matrix = np.sqrt(deltas_hat_matrix)
        # compute mean of sqrt(delta_hat) for each k
        s_bar = np.mean(sqrt_deltas_hat_matrix, axis=0) # row vector
        # compute vector a_ik - a_bar_k, where a_ik = sqrt(delta_ik)*y_i
        As = [self.y * sqrt_deltas_hat_matrix[:,k].reshape(-1,1) for k in range(self.N_classes)]
        a_bars = [np.mean(Ak, axis=0) for Ak in As]
        for _,Ak,a_bar_k in zip(range(self.N_classes),As,a_bars): # center the array and verticalize a_bar
            Ak -= np.ones_like(Ak) * a_bar_k # NOTE: using "-=" updates the list :)
            a_bars[_] = a_bar_k.reshape(-1,1)
        # cov matrix os the Aks
        Sas = [Ak.T @ Ak / self.N for Ak in As]
        # centered vector of sqrt(delta_hat)
        ds = [sqrt_deltas_hat_matrix[:,k].reshape(-1,1) - np.ones((self.N,1))*s_bar[k] for k in range(self.N_classes)]
        # factor (d_k^T @ d_k) to be used to compute Sbk during M-step
        # (we cannot store Sbk directly because it depends on beta, which is a free param in M-step)
        dtds = [dk.T @ dk for dk in ds]
        # factor (Ak^T @ dk) to be used to compute the cross-cov matrix Sab during M-step
        # (again, we cannot compute Sab directly because it also dpeends on beta)
        Atds = [Ak.T @ dk for Ak, dk in zip(As,ds)]
        return deltas_hat_matrix, p, s_bar, a_bars, Sas, dtds, Atds

    def minus_Q_no_pis(self, theta, *args):
        """objective function to minimize in the case step_M_per_class=False (i.e. we fit the params
           of all classes at the same time during M-step of the EM algo). This is the M-step log-likelihood.
           The "no_pis" in the name means that the classes probabilities are explicitly estimated, so they won't
           impact the minimization of this objective function

        Args:
            theta (1D array): parameter vector as received by self.parse_theta
                              (but without the pis, otherwise there is no minimum...)

        Returns:
            scalar: log-likelihood for M-step (not considering the classes probabilities pis)
        """
        Rs, betas = self.parse_theta(theta, pis_included=False)
        _, p, s_bar, a_bars, Sas, dtds, Atds = args
        # fast way of inverting R (which is either multiple of identity of just diagonal)
        inv_Rs = [np.eye(self.T) * 1/np.diag(R) for R in Rs]
        # fast way of computing determinant of R
        det_Rs = [np.diag(R).prod() for R in Rs]
        return - (
            -1/2 * sum([
                    self.N * np.trace(inv_Rs[k] @ Sas[k]) +
                    np.trace(inv_Rs[k] @ (self.X @ betas[k]) @ dtds[k] @ (self.X @ betas[k]).T) +
                    self.N * (a_bars[k]-s_bar[k]*(self.X@betas[k])).T @ inv_Rs[k] @ (a_bars[k]-s_bar[k]*(self.X@betas[k])) -
                    np.trace(inv_Rs[k] @ Atds[k] @ (self.X @ betas[k]).T) -
                    np.trace(inv_Rs[k] @ (self.X @ betas[k]) @ Atds[k].T) +
                    p[k] * np.log(det_Rs[k])
                    for k in range(self.N_classes)
                ])
            - self.N * self.T / 2 * np.log(2*np.pi)
            # + sum([
            #     sum(
            #         [deltas_hat_matrix[i,k] * np.log(pis[k]) for i in range(self.N)]
            #         )
            #     for k in range(self.N_classes)
            #     ])
        )[0,0] # the expression results in a 1x1 matrix, but we want to return a scalar

    def class_M_step_obj(self, theta, *args):
        """objective function to minimize in the case step_M_per_class=True (i.e. we fit the params
           of one class at a time time during M-step of the EM algo).

        Args:
            theta (1D array): parameter vector as received by self.parse_theta_one_class
                              (but without the pi, otherwise there is no minimum...)

        Returns:
            scalar: log-likelihood for M-step (not considering the class probability pi)
        """
        R, beta = self.parse_theta_one_class(theta, pi_included=False)
        pk, s_bark, a_bar, Sa, dtd, Atd = args # NOTE 1 less element than *args in self.minus_Q_no_pis
        # fast way of inverting R (which is either multiple of identity of just diagonal)
        inv_R = np.eye(self.T) * 1/np.diag(R)
        # fast way of computing determinant of R
        det_R = np.diag(R).prod()
        return - (
            -1/2 * (
                    self.N * np.trace(inv_R @ Sa) +
                    np.trace(inv_R @ (self.X @ beta) @ dtd @ (self.X @ beta).T) +
                    self.N * (a_bar-s_bark*(self.X@beta)).T @ inv_R @ (a_bar-s_bark*(self.X@beta)) -
                    np.trace(inv_R @ Atd @ (self.X @ beta).T) -
                    np.trace(inv_R @ (self.X @ beta) @ Atd.T) +
                    pk * np.log(det_R)
                )
            - self.N * self.T * np.log(2*np.pi) / (2 * self.N_classes)
        )[0,0] # the expression results in a 1x1 matrix, but we want to return a scalar

    def get_clusterwise_probabilities(self):
        assert self.deltas_hat_final is not None, "probs of each subject belonging to each class called before fitting"
        return np.copy(self.deltas_hat_final)

    def get_predictions(self):
        assert self.predicitons is not None, "predictions of most likely cluster called before fitting"
        return np.copy(self.predicitons)

    def solve(self, nrep=10, verbose=True, step_M_per_class=True):
        """fit the LCGA classes

        Args:
            nrep (int) : number of multistart repetitions. Defaults to 10.
            verbose (bool, optional): Verbose mode. Defaults to True.
            step_M_per_class (bool, optional): Whether to fit each class at a time in M-step.
                                               (Faster if True, and theoretically the same results,
                                               but option False kept just in case)
                                               Defaults to True.

        Returns:
            (list, list, list): Rs, betas, pis
        """

        n_params_R = 1 if self.R_struct == 'multiple_identity' else self.T

        theta_opt = None # optimal parameter vector (lowest loglik)
        E_opt = None # values of the E-step for the optimal repetition
        opt_val = np.Inf # optimal loglik
        opt_val_list = [] # the final loglik for each repetition

        ########### MULTISTART ############
        for _ in range(nrep):
            #### Initialization ####
            theta0 = np.zeros(self.N_classes*(n_params_R + self.k + 1))
            # initialize the Rs
            theta0[0:n_params_R*self.N_classes] = np.random.rand()*min([np.sqrt(np.var(self.y[:,t])) for t in range(self.T)]) # standard dev of measures in first time step
            # intialize the betas
            samples_idxs = np.random.choice(np.arange(self.N), size=self.N_classes, replace=False)
            for k in range(self.N_classes):
                sample_y = self.y[samples_idxs[k]].reshape(-1,1)
                theta0[n_params_R*self.N_classes+k*self.k:n_params_R*self.N_classes+(k+1)*self.k] = (
                        np.linalg.inv(self.X.T @ self.X) @ self.X.T @ sample_y # initialize with linear regression
                    ).flatten()
            # initialize the pis
            pis_0 = np.random.rand(self.N_classes)
            pis_0 /= np.sum(pis_0)
            theta0[-self.N_classes:] = pis_0
            # initialize the auxiliary parameter vector for the E-M loop
            theta_prev = -1 * np.ones(self.N_classes*(n_params_R + self.k + 1))
            counter = 0
            #### EM Algorithm ####
            while np.linalg.norm(theta_prev - theta0) > 1E-8 and counter < 500:
                theta_prev = np.copy(theta0)
                # E-step:
                E = self.E_step(theta0)
                # M-step:
                # first, fit the pis:
                pis_opt = E[1] / self.N
                theta0[-self.N_classes:] = pis_opt
                # then, the other parameters:
                if step_M_per_class:
                    _, p, s_bar, a_bars, Sas, dtds, Atds = E
                    off_R = self.N_classes*n_params_R # offset, theta is in the order: Rs, and then betas
                    for k in range(self.N_classes):
                        vals = (p[k], s_bar[k], a_bars[k], Sas[k], dtds[k], Atds[k])
                        input = np.concatenate(
                            (theta0[k*n_params_R:(k+1)*n_params_R], theta0[off_R+k*self.k:off_R+(k+1)*self.k])
                        )
                        optimize_res = minimize(self.class_M_step_obj, input, args=vals, jac='3-point', options={'disp':False})
                        theta0[k*n_params_R:(k+1)*n_params_R] = optimize_res.x[0:n_params_R]
                        theta0[off_R+k*self.k:off_R+(k+1)*self.k] = optimize_res.x[n_params_R:]
                        if verbose:
                            print('{}-th EM iteration, class {}: {}'.format(counter, k,
                            'success' if optimize_res.success else 'failed...'))
                else:
                    optimize_res = minimize(self.minus_Q_no_pis, theta0[0:-self.N_classes], args=E,
                        jac='3-point', options={'disp':False})
                    theta0[:-self.N_classes] = optimize_res.x
                    if verbose:
                        print('{}-th EM iteration: {}; eval = {}'.format(counter,
                        'success' if optimize_res.success else 'failed...',
                        optimize_res.fun))
                counter += 1
            if verbose:
                print() # break line
            fun = - sum(np.log(pis_opt) * E[1]) # (-1) * sum_k [ log(pi) * sum_i[delta_hat] ]
            if step_M_per_class:
                for k in range(self.N_classes):
                    vals = (p[k], s_bar[k], a_bars[k], Sas[k], dtds[k], Atds[k])
                    theta0_class = np.concatenate(
                        (theta0[k*n_params_R:(k+1)*n_params_R], theta0[off_R+k*self.k:off_R+(k+1)*self.k])
                    )
                    fun += self.class_M_step_obj(theta0_class, *vals)
            else:
                fun += self.minus_Q_no_pis(theta0[0:-self.N_classes], *E)
            opt_val_list.append(fun)
            if fun < opt_val:
                opt_val = fun
                theta_opt = theta0
                E_opt = E

        if verbose:
            print("Optimal value per repetiton with random initialization: {}".format(opt_val_list))

        # other information (to be returned by auxiliary methods)
        # 1- probability, for each subject, of belonging to each class
        self.deltas_hat_final = E_opt[0]
        # 2- most likely class for each subject
        self.predicitons = np.argmax(self.deltas_hat_final, axis=1).astype('int')

        return self.parse_theta(theta_opt, pis_included=True)
