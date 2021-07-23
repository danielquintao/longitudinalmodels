import numpy as np
from models.GCM import GCM
from models.LCGA import LCGA
from utils.gcm_plot import plot, extended_plot
from utils.lcga_plot import plot_lcga
from utils.convert_data import convert_label
from harmonizer.Harmonizer import Harmonizer

def run_pipeline_GCM(y_main, timesteps, max_degree, y_control=None, src_labels1D=None, R_struct='multiple_identity'):

    # 0- apply transformation to input if too big of too small
    # if not ( 1 < max(timesteps) - min(timesteps) < 10):
    #     print('Your timesteps seem to be in either a too short or too wide scale.')
    #     val = None
    #     while val is None:
    #         print('Would you like to redimensionate your time steps to the interval [0,1]? y/n')
    #         print('(Knowing that ALL the following results will need to be interpreted accordingly)')
    #         val = input()
    #         if val in ['y','Y','yes','Yes','YES']:
    #             m, M = min(timesteps), max(timesteps)
    #             timesteps = [(t-m)/(M-m) for t in timesteps]
    #             print('OK! You new timesteps are: {}'.format(timesteps))
    #         elif val in ['n','N','no','No','NO']:
    #             print('Alright, we will keep the original scale')
    #         else:
    #             print('Please type y (for yes) or n (for no)')
    #             val = None
    
    # 1- remove outliers and basic treatment
    # TODO

    # 2- assertion harmonization model
    assert (
        (y_control is not None and src_labels1D is not None) or
        (y_control is None and src_labels1D is None)
    ), "in order to use harmonize data, both y_control and src_labels1D must be provided"

    # 3- GCM
    print("==========================================================")
    print("      Growth Curve Model Analysis - no groups             ")
    print("==========================================================\n")
    for degree in range(max_degree, 0, -1):
        if degree >= len(timesteps)-1:
            print('degree {} ignored: We recommend to use only degrees lower than #timesteps - 1'.format(degree))
            continue
        print('degree {}-\n'.format(degree))
        #harmonize data:
        if y_control is not None:
            harmonizer = Harmonizer(y_main, y_control, src_labels1D, timesteps, degree)
            print("harmonizing data")
            _,_ = harmonizer.fit(verbose=False)
            y,_ = harmonizer.transform(y_main, y_control, src_labels1D)
            print("data harmonized")
        else:
            y = np.copy(y_main)
        # apply model:
        gcm = GCM(y, timesteps, degree, R_struct)
        try:
            beta_opt, R_opt, D_opt = gcm.solve(verbose=False)
        except AssertionError as err:
            print('something went wrong while fitting the model:')
            print(err)
            print()
            continue
        print('Fixed effects:')
        print(beta_opt.reshape(-1,1))
        print('Random effects covariance matrix:')
        print(D_opt)
        print('Residual deviations covariance matrix:')
        print(R_opt)
        print('Extra information - random effects correlation matrix:')
        d = np.sqrt(np.diag(D_opt))
        corr_D = D_opt / (d * d.reshape(-1,1))
        print(corr_D)
        # big_corr = np.tril(abs(corr_D) > 0.8, k=-1)
        # if np.any(big_corr):
        #     print("The following pairs of coefficients have correlation greater than 0.8:")
        #     print('(0 stands for the intercept, 1 for the slope, etc)')
        #     print([(k,l) for k,l in np.argwhere(big_corr)[:]])
        plot(beta_opt, timesteps, y, degree)
        print()

def run_pipeline_extended_GCM(y, timesteps, max_degree, R_struct='multiple_identity', groups=None):

    if groups is None:
        print("no groups detected! automatically calling pipeline for normal GCM...")
        run_pipeline_GCM(y, timesteps, R_struct)
        return
    print("==========================================================")
    print("      Growth Curve Model Analysis - with groups           ")
    print("==========================================================\n")
    print('We will now take the known groups into consideration for the fixed effects, '+
          'but we still assume that all the subjects, of all groups, share the random effects matrix '+
          'and the residual deviation matrix')
    # converting groups format:
    plot_with_categorical = False
    if ((len(groups.shape) == 1 or (len(groups.shape) == 2 and groups.shape[1] == 1))
        and not all([g in [0,1] for g in groups])): # need to convert to one-hot
        print('Warning: We need to convert the groups from the categorical representation to a custom one-hot representation')
        print('(e.g. groups 1,2,3,4 would become (0,0,0),(1,0,0),(0,1,0),(0,0,1) resp.\n')
        if not np.issubdtype(groups.dtype, np.int):
            groups = groups.astype('int')
        groups_converted = convert_label(groups, offset=min(groups))
        plot_with_categorical = True
    else: # already received groups in good format
        groups_converted = groups
    # estimations:
    for degree in range(max_degree, 0, -1):
        if degree >= len(timesteps)-1:
            print('degree {} ignored: We recommend to use only degrees lower than #timesteps - 1'.format(degree))
            continue
        print('degree {}-\n'.format(degree))
        gcm = GCM(y, timesteps, degree, R_struct, groups_converted)
        try:
            betas_opt, R_opt, D_opt = gcm.solve(verbose=False)
        except AssertionError as err:
            print('something went wrong while fitting the model:')
            print(err)
            continue
        print('Fixed effects:')
        beta = betas_opt[0:degree+1].reshape(-1,1)
        print(beta)
        for i in range(1,groups_converted.shape[1]+1):
            beta_i = betas_opt[0:degree+1] + betas_opt[i*(degree+1):(i+1)*(degree+1)]
            print(beta_i.reshape(-1,1))
        print('Random effects covariance matrix:')
        print(D_opt)
        print('Residual deviations covariance matrix:')
        print(R_opt)
        if plot_with_categorical:
            plot_lcga(gcm.pretty_beta(betas_opt), timesteps, y, degree, groups-min(groups))
        else:
            groups2plot = list(set([tuple(r.astype(int)) for r in groups_converted[:]]))
            extended_plot(betas_opt, timesteps, y, groups_converted, groups2plot, degree)
        print()

def run_pipeline_LCGA(y, timesteps, max_degree, R_struct='multiple_identity', max_latent_classes=3):

    print("==========================================================")
    print("      Latent Class Growth Analysis                        ")
    print("==========================================================\n")
    for degree in range(max_degree, 0, -1):
        if degree >= len(timesteps)-1:
            print('degree {} ignored: We recommend to use only degrees lower than #timesteps - 1'.format(degree))
            continue
        for K in range(max_latent_classes,1,-1):
            print('degree {}, {} latent classes-\n'.format(degree, K))
            lcga = LCGA(y, timesteps, degree, K)
            try:
                print('estimating latent classes...')
                Rs, betas, pis= lcga.solve(verbose=False)
            except AssertionError as err:
                print('something went wrong while fitting the model:')
                print(err)
                continue
            for i in range(K):
                print('cluster {}:'.format(i))
                print('Fixed effects:')
                print(betas[i])
                print('Residual deviations covariance matrix:')
                print(Rs[i])
                print()
            preds = lcga.get_predictions()
            plot_lcga(betas, timesteps, y, degree, preds)
            print()