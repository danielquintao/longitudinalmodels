import numpy as np
from scipy.optimize import minimize
from utils.gcm_plot import extended_plot

class LCGA():
    def __init__(self, y, timesteps, degree, N_classes, R_struct="multiple_identity"):
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
        if self.R_struct == 'multiple_identity':
            R = np.eye(self.T) * theta[0] ** 2
            left = 1
        else:
            R = np.eye(self.T) * theta[0:self.T] ** 2
            left = self.T
        betas = []
        for _ in range(self.N_classes):
            right = left + self.k
            betas.append(theta[left:right].reshape(-1,1))
            left = right
        if pis_included:
            pis = [theta[left + _] for _ in range(self.N_classes)]
            return R, betas, pis
        return R, betas

    def E_step(self, theta):
        R, betas, pis = self.parse_theta(theta, pis_included=True)
        # compute delta_hat's
        deltas_hat_matrix = np.zeros((self.N, self.N_classes))
        for i in range(self.N):
            for k in range(self.N_classes):
                yi = self.y[i].reshape(-1,1)
                deltas_hat_matrix[i,k] = pis[k] * self.multivar_normal_PDF(yi, R, betas[k])
            # the above expression of sqrt(delta_hat) is incomplete: we must normalize and take the square root
            if sum(deltas_hat_matrix[i,:]) != 0: # deal with numerical approx -> small vals are stored as 0
                deltas_hat_matrix[i,:] /= sum(deltas_hat_matrix[i,:])
            else:
                deltas_hat_matrix[i,:] = 1./self.N_classes # necessary HACK !!!
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
        return deltas_hat_matrix, s_bar, a_bars, Sas, dtds, Atds

    def minus_Q_no_pis(self, theta, *args):
        R, betas = self.parse_theta(theta, pis_included=False)
        _, s_bar, a_bars, Sas, dtds, Atds = args
        # fast way of inverting R (which is either multiple of identity of just diagonal)
        inv_R = np.eye(self.T) * 1/np.diag(R)
        # fast way of computing determinant of R
        det_R = np.diag(R).prod()
        return - (
            -1/2 * sum([
                    self.N * np.trace(inv_R @ Sas[k]) +
                    np.trace(inv_R @ (self.X @ betas[k]) @ dtds[k] @ (self.X @ betas[k]).T) +
                    self.N * (a_bars[k]-s_bar[k]*(self.X@betas[k])).T @ inv_R @ (a_bars[k]-s_bar[k]*(self.X@betas[k])) +
                    np.trace(inv_R @ Atds[k] @ (self.X @ betas[k]).T) +
                    np.trace(inv_R @ (self.X @ betas[k]) @ Atds[k].T)
                    for k in range(self.N_classes)
                ])
            - self.N * self.T / 2 * np.log(2*np.pi)
            - self.N / 2 * np.log(det_R)
            # + sum([
            #     sum(
            #         [deltas_hat_matrix[i,k] * np.log(pis[k]) for i in range(self.N)]
            #         )
            #     for k in range(self.N_classes)
            #     ])
        )[0,0] # the expression results in a 1x1 matrix, but we want to return a scalar

    def solve(self):

        n_params_R = 1 if self.R_struct == 'multiple_identity' else self.T

        # Initialization
        theta0 = np.zeros(n_params_R + self.N_classes*self.k + self.N_classes)
        # initialize R
        theta0[0:n_params_R] = np.var(self.y[:,0]) # variance of measures in first time step
        # intialize the betas
        samples_idxs = np.random.choice(np.arange(self.N), size=self.N_classes, replace=False)
        for k in range(self.N_classes):
            sample_y = self.y[samples_idxs[k]].reshape(-1,1)
            theta0[n_params_R+k*self.k:n_params_R+(k+1)*self.k] = (
                    np.linalg.inv(self.X.T @ self.X) @ self.X.T @ sample_y # initialize with linear regression
                ).flatten()
        # initialize the pis
        pis_0 = np.random.rand(self.N_classes)
        pis_0 /= np.sum(pis_0)
        theta0[-self.N_classes:] = pis_0
        # initialize the auxiliary parameter vector for the E-M loop
        theta_prev = -1 * np.ones(n_params_R + self.N_classes*self.k + self.N_classes)
        counter = 0

        while np.linalg.norm(theta_prev - theta0) > 1E-8 and counter < 500:
            theta_prev = np.copy(theta0)
            # E-step:
            E = self.E_step(theta0)
            # M-step:
            # first, fit the pis
            deltas_hat_matrix = E[0]
            pis_opt = np.sum(deltas_hat_matrix, axis=0) / np.sum(deltas_hat_matrix)
            # then, the other parameters
            optimize_res = minimize(self.minus_Q_no_pis, theta0[0:-self.N_classes], args=E,
                jac='3-point', options={'disp':False}) ## FIXME 'disp' = verbose?
            theta0[:-self.N_classes] = optimize_res.x
            theta0[-self.N_classes:] = pis_opt
            counter += 1

        return self.parse_theta(theta0, pis_included=True)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ######### TEST PDF #########

    # total_data = np.genfromtxt("test/playground_data/benchmark1_data.csv", delimiter=",", skip_header=0)
    # # TRUNCATE TO 2 TIME STEPS:
    # y = total_data[:,0:1] # measures in time steps
    # time = np.array([0., 0.5]) # cf. benchmark1_ground_truth.txt
    # degree = 0 # cf. benchmark1_ground_truth.txt
    # N_classes = 2
    # model = LCGA(y, time, degree, N_classes)

    # # simulate multivariate normal data and check function that computes PDF
    # beta = np.random.rand(degree+1,1)
    # mean = (model.X @ beta).flatten()
    # data = np.random.multivariate_normal(mean, np.eye(len(time)), 5000)
    # PDFs = []
    # for datum in data:
    #     val = model.multivar_normal_PDF(datum.reshape(-1,1), np.eye(len(time)), beta)
    #     assert 0 <= val <= 1, "OOPS"
    #     PDFs.append(val)
    # plt.figure()
    # plt.scatter(data[:,0], data[:,1], s=2, c=np.array(PDFs))
    # plt.show()

    ######### TEST OTHER STUFF #########

    total_data = np.genfromtxt("test/playground_data/benchmark5_data.csv", delimiter=",", skip_header=0)
    y = total_data[:,0:4] # measures in time steps
    time = np.array([0., 2., 4., 6.]) # cf. benchmark1_ground_truth.txt
    degree = 1 # cf. benchmark1_ground_truth.txt
    N_classes = 2
    model = LCGA(y, time, degree, N_classes)

    # # test E_step, minus_Q (ideally in debug mode of an IDE)
    # E = model.E_step(np.random.rand(1 + len(time) + N_classes * (degree+1)))
    # res = model.minus_Q(np.random.rand(1 + len(time) + N_classes * (degree+1)), *E)
    # print(res)

    R, betas, pis = model.solve()
    eta = np.concatenate((betas[0],betas[1]), axis=1).flatten()
    extended_plot(eta, time, y, np.zeros((len(y),1)), [(0,),(1,)], 1)