import numpy as np
from matrix_utils import flattened2triangular
from gcm_plot import extended_plot, plot
import matplotlib.pyplot as plt

def generate_sample(N, time, degree, N_groups, output_file=None, scaling=None):
    """generate fake longitudinal sample
       Note: We have a fixed scale for generating the covariance matrices (covariances are
             generated from Cholesky decompositins with elements in range [0,1)). Hence, in
             order to control the general appearence of the dataset, you may want to work
             with both the time steps scale (e.g. choosing smaller or bigger steps) and with
             the optional variable scaling. Don't hesitate to generate a plot of the data.

    Args:
        N (int): desired number of individuals
        time (1D numpy array): timesteps
        degree (int): degree of the underlying polynomial model
        N_groups (int): number of groups to which individuals belong; these groups will be modeled
                        as bit vectors of size N_groups-1, following the convention (0,0,..,0) for
                        the a group, then (1,0,0..0), (0,1,0..0), (0,0,1..0), ... for the others
        output_file (str): path to the file (includes name) where output will be stored. None if we
                           choose not to generate output. Suffixes "_data" and "_ground_truth"
                           will be added to the file name, and two files will be created (one with
                           the data, the other one with the underlying curve parameters). Defaults
                           to None.
        scaling (1D numpy array (len degree+1), optional): scale to apply to the m-th order coeff
                                                           of the equations underlying each group,
                                                           which are independently generated in the 
                                                           interval [0,1) with numpy.rand().
                                                           Defaults to None (scale '1' for every
                                                           coefficient order).

    Returns:
        [type]: [description]
    """
    # /!\ the convention of the var name N_groups here is different from the code of the solver
    # set scaling
    if scaling is None:
        scaling = np.ones(degree+1)
    # generate beta
    beta = []
    for i in range(N_groups):
        beta.extend(list(scaling * np.random.rand(degree+1)))
    beta = np.array(beta).reshape(-1,1)
    # generate covarianve matrixes (using a "cholesky trick" to ensure properties)
    T = len(time)
    k = degree+1
    R_upper = flattened2triangular(np.random.rand(int(T*(T+1)/2)))
    R = R_upper.T @ R_upper
    D_upper = flattened2triangular(np.random.rand(int(k*(k+1)/2)))
    D = D_upper.T @ D_upper
    # generate matrix Z of shape e.g. [[1,t0,t0^2],[1,t1,t1^2]]
    Z = np.ones((T,1))
    time = np.array(time).reshape(-1,1)
    for i in range(1, degree+1):
        Z = np.concatenate((Z,time**i), axis=1)
    # generate random individuals
    data = np.zeros((N,T+(N_groups-1)))
    for i in range(N):
        # First, we'll assign it a random group
        g = np.random.choice(np.arange(N_groups))
        # We'll encode each group with a bit vector of length N_groups-1
        # The convention is: (0,0,..,0) for first group, then (1,0,0..0), (0,1,0..0), (0,0,1..0) ...
        # Why this choice? I don't know but it seems not bad
        bit_vec_g = np.zeros(N_groups-1) # n groups encodeded with bit vectors of length n-1
        if g > 0:
            bit_vec_g[g-1] = 1
        # Build X from Z
        X = Z.copy()
        for j in range(N_groups-1):
            X = np.concatenate((X,bit_vec_g[j]*Z),axis=1)
        # Generate fake observations from the groups curve, R and D
        cov_mat = R + Z @ D @ Z.T
        mu = (X @ beta).flatten()
        yi = np.random.multivariate_normal(mu,cov_mat)
        data[i] = np.concatenate((yi,bit_vec_g))
    # persist output
    if output_file is not None:
        str_format = []
        for i in range(T):
            str_format.append('%.8f')
        for i in range(N_groups-1):
            str_format.append('%d')
        if output_file[-4:] ==  '.txt' or output_file[-4:] ==  '.csv':
            output_file = output_file[:-4]
        np.savetxt(output_file+"_data.csv", data, fmt=str_format, delimiter=",")
        other_file = open(output_file+"_ground_truth.txt", 'w')
        other_file.write('degree:\n{}\n\n'.format(degree))
        other_file.write('time-points:\n{}\n\n'.format(time))
        other_file.write('number of groups:\n{}\n\n'.format(N_groups))
        other_file.write('beta:\n{}\n\n'.format(beta.flatten()))
        other_file.write('R:\n{}\n\n'.format(R))
        other_file.write('D:\n{}\n'.format(D))
        other_file.close()
    return data, beta.flatten(), R, D

time = np.array([0,0.5,1,1.5])
degree = 1
data, beta, R, D = generate_sample(50, time, degree, 1, output_file="playground_data/benchmark2", scaling=[1,1])
# print(data,'\n',beta,'\n',R,'\n',D)
# extended_plot(beta, time, data[:,0:4], data[:,-1:], [(0,),(1,)], degree) # P.S. (x,) -> "singleton" tuple        
plot(beta, time, data, degree)

        

