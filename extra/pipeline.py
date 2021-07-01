import numpy as np
from models.GCM import GCM
from models.LCGA import LCGA
from utils.gcm_plot import plot, extended_plot
from utils.lcga_plot import plot_lcga

def run_pipeline(y, timesteps, max_degree, R_struct='multiple_identity', groups=None):
    
    # 1- remove outliers and basic treatment
    # TODO

    # 2- call harmonization model
    # TODO

    # 3- GCM
    print("==========================================================")
    print("      Growth Curve Model Analysis - no groups             ")
    print("==========================================================\n")
    for degree in range(max_degree, 0, -1):
        if degree >= len(timesteps)-1:
            print('degree {} ignored: We recommend to use only degrees lower than #timesteps - 1'.format(degree))
            continue
        print('degree {}-\n'.format(degree))
        gcm = GCM(y, timesteps, degree, R_struct)
        try:
            beta_opt, R_opt, D_opt = gcm.solve(verbose=False)
        except AssertionError as err:
            print('something went wrong while fitting the model:')
            print(err)
            continue
        print('Fixed effects:')
        print(beta_opt.reshape(-1,1))
        print('Random effects covariance matrix:')
        print(D_opt)
        print('Residual deviations covariance matrix:')
        print(R_opt)
        plot(beta_opt, timesteps, y, degree)
        
