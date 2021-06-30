import numpy as np
from scipy.optimize import minimize
from utils.lcga_plot import plot_lcga_TWO_groups, plot_lcga

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

    def get_clusterwise_probabilities(self):
        assert self.deltas_hat_final is not None, "probs of each subject belonging to each class called before fitting"
        return np.copy(self.deltas_hat_final)

    def get_predictions(self):
        assert self.predicitons is not None, "predictions of most likely cluster called before fitting"
        return np.copy(self.predicitons)

    def solve(self, verbose=True):

        n_params_R = 1 if self.R_struct == 'multiple_identity' else self.T

        # Initialization
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

        while np.linalg.norm(theta_prev - theta0) > 1E-8 and counter < 500:
            theta_prev = np.copy(theta0)
            # E-step:
            E = self.E_step(theta0)
            # M-step:
            # first, fit the pis:
            pis_opt = E[1] / self.N
            # then, the other parameters:
            optimize_res = minimize(self.minus_Q_no_pis, theta0[0:-self.N_classes], args=E,
                jac='3-point', options={'disp':False})
            theta0[:-self.N_classes] = optimize_res.x
            theta0[-self.N_classes:] = pis_opt
            if verbose:
                print('{}-th EM iteration: {}; eval = {}'.format(counter,
                'success' if optimize_res.success else 'failed...',
                optimize_res.fun))
            counter += 1

        # other information to be returned
        # 1- probability, for each subject, of belonging to each class
        self.deltas_hat_final = E[0]
        # 2- most likely class for each subject
        self.predicitons = np.argmax(self.deltas_hat_final, axis=1).astype('int')

        return self.parse_theta(theta0, pis_included=True)
