import numpy as np
from models.GCM import GCM
from models.LCGA import LCGA
from utils.gcm_plot import plot, extended_plot
from utils.lcga_plot import plot_lcga
from utils.model_selection_plot import plot_loglikelihoods, plot_information_criterions
from utils.convert_data import convert_label
from harmonizer.Harmonizer import Harmonizer
from harmonizer.Harmonizer_extended import ExtendedHarmonizer
from scipy.stats import chi2
from itertools import combinations

def run_pipeline_GCM(y_main, timesteps, max_degree, y_control=None, src_labels1D=None,
    R_struct='multiple_identity', varname=None, use_log=False):

    # 0- remove outliers and basic treatment
    # TODO

    # 1- basic assertions
    assert (
        (y_control is not None and src_labels1D is not None) or
        (y_control is None and src_labels1D is None)
    ), "in order to use harmonize data, both y_control and src_labels1D must be provided"
    assert not np.any(np.isnan(y_main)), "NaN (Not A Number) detected in y_main"
    assert not np.any(np.isinf(np.abs(y_main))), "Inf or -Inf detected in y_main"
    if y_control is not None:
        assert not np.any(np.isnan(y_control)), "NaN (Not A Number) detected in y_control"
        assert not np.any(np.isinf(np.abs(y_control))), "Inf or -Inf detected in y_control"

    # 2- log?
    if use_log:
        if np.any(y_main <= 0):
            print("We found negative values in y_main and the log transformation was NOT applied.\n" +
                "We maintained the original scale.\n") 
        elif y_control is not None and np.any(y_control <= 0):
            print("We found negative values in y_control and the log transformation was NOT applied.\n" +
                "We maintained the original scale.\n") 
        else:
            y_main = np.log10(y_main)
            y_control = np.log10(y_control) if y_control is not None else None
            print("We converted data to their logarithm (in base 10)")

    # 3- GCM
    print("==========================================================")
    print("      Growth Curve Model Analysis - no groups             ")
    print("==========================================================\n")
    for degree in range(max_degree, 0, -1):
        if degree >= len(timesteps)-1:
            print('degree {} ignored: We recommend to use only degrees lower than #timesteps - 1'.format(degree))
            continue
        print('degree {}-\n'.format(degree))
        # harmonize data:
        if y_control is not None:
            harmonizer = Harmonizer(y_main, y_control, src_labels1D, timesteps, degree)
            print("harmonizing data")
            try:
                _,_ = harmonizer.fit(verbose=False, force_solver=True)
            except AssertionError as err:
                print('something went wrong while harmonizing the data:')
                print(err)
                print("We will use the original data")
                y = np.copy(y_main)
                print()
            else:
                y,_ = harmonizer.transform(y_main, y_control, src_labels1D)
                print("data harmonized")
        else:
            y = np.copy(y_main)
        # apply model:
        gcm = GCM(y, timesteps, degree, R_struct)
        try:
            beta_opt, R_opt, D_opt = gcm.solve(verbose=False)
        except Warning as war:
            print(war) # just print and go on
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
        df1, df2 = gcm.degrees_of_freedom()
        df = df1 + df2
        n_params = gcm.get_n_params()
        loglik = gcm.get_loglikelihood()
        print("loglik = {}, df = {}, nb params = {}, AIC = {}, BIC = {}".format(loglik, df, n_params, 2*(n_params-loglik), np.log(len(y))*n_params-2*loglik))
        varname = '$log_{10}$('+varname+')' if (use_log and varname is not None) else varname
        plot(beta_opt, timesteps, y, degree, title='GCM - degree {}'.format(degree), varname=varname)
        print()

def run_pipeline_extended_GCM(y_main, timesteps, max_degree, groups=None,
    y_control=None, src_labels1D=None, R_struct='multiple_identity', varname=None, use_log=False):

    # 0- remove outliers and basic treatment
    # TODO

    # 1- basic assertions
    assert (
        (y_control is not None and src_labels1D is not None) or
        (y_control is None and src_labels1D is None)
    ), "in order to use harmonize data, both y_control and src_labels1D must be provided"
    if groups is None:
        print("no groups detected! automatically calling pipeline for normal GCM...")
        run_pipeline_GCM(y_main, timesteps, max_degree, y_control, src_labels1D, R_struct, varname)
        return
    assert not np.any(np.isnan(y_main)), "NaN (Not A Number) detected in y_main"
    assert not np.any(np.isinf(np.abs(y_main))), "Inf or -Inf detected in y_main"
    if y_control is not None:
        assert not np.any(np.isnan(y_control)), "NaN (Not A Number) detected in y_control"
        assert not np.any(np.isinf(np.abs(y_control))), "Inf or -Inf detected in y_control"

    # 2- log?
    if use_log:
        if np.any(y_main <= 0):
            print("We found negative values in y_main and the log transformation was NOT applied.\n" +
                "We maintained the original scale.\n") 
        elif y_control is not None and np.any(y_control <= 0):
            print("We found negative values in y_control and the log transformation was NOT applied.\n" +
                "We maintained the original scale.\n") 
        else:
            y_main = np.log10(y_main)
            y_control = np.log10(y_control) if y_control is not None else None
            print("We converted data to their logarithm (in base 10)")

    # 3 - GCM with groups
    print("==========================================================")
    print("      Growth Curve Model Analysis - with groups           ")
    print("==========================================================\n")
    print('We assume that all the subjects, of all groups, share the random effects matrix '+
          'and the residual deviation matrix')
    # converting groups format:
    if not np.issubdtype(groups.dtype, np.int):
        groups = groups.astype('int')
    plot_with_categorical = False
    groups = groups.reshape(-1,1) if len(groups.shape) == 1 else groups
    if (groups.shape[1] == 1 and not all([g in [0,1] for g in groups])): # need to convert to one-hot
        print('Note: We will convert the groups from the categorical representation to a custom one-hot representation')
        print('(e.g. groups 1,2,3,4 would become (0,0,0),(1,0,0),(0,1,0),(0,0,1) resp.\n')
        groups_converted = convert_label(groups, offset=np.min(groups,axis=None))
        plot_with_categorical = True
    else: # already received groups in good format
        groups_converted = groups
        # NOTE: attention to distinguish groups_converted from groups in the code below
    # estimations:
    for degree in range(max_degree, 0, -1):
        if degree >= len(timesteps)-1:
            print('degree {} ignored: We recommend to use only degrees lower than #timesteps - 1'.format(degree))
            continue
        print('degree {}-\n'.format(degree))
        # harmonize data:
        if y_control is not None:
            harmonizer = ExtendedHarmonizer(y_main, y_control, src_labels1D, groups_converted,timesteps, degree)
            print("harmonizing data")
            try:
                _,_ = harmonizer.fit(verbose=False, force_solver=True)
            except AssertionError as err:
                print('something went wrong while harmonizing the data:')
                print(err)
                print("We will use the original data")
                y = np.copy(y_main)
                print()
            else:
                y,_ = harmonizer.transform(y_main, y_control, src_labels1D, groups_converted)
                print("data harmonized")
        else:
            y = np.copy(y_main)
        # apply model:
        gcm = GCM(y, timesteps, degree, R_struct, groups_converted)
        try:
            betas_opt, R_opt, D_opt = gcm.solve(verbose=False) # XXX it seems dum not to use betas_pretty=True, but extended_plot uses the ugly beta format
        except Warning as war:
            print(war) # just print and go on
        except AssertionError as err:
            print('something went wrong while fitting the model:')
            print(err)
            continue
        print('Fixed effects for each group:')
        beta = betas_opt[0:degree+1].reshape(-1,1)
        print(beta)
        for i in range(1,groups_converted.shape[1]+1):
            beta_i = betas_opt[0:degree+1] + betas_opt[i*(degree+1):(i+1)*(degree+1)]
            print(beta_i.reshape(-1,1))
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
        df1, df2 = gcm.degrees_of_freedom()
        df = df1 + df2
        n_params = gcm.get_n_params()
        loglik = gcm.get_loglikelihood()
        print("loglik = {}, df = {}, nb params = {}, AIC = {}, BIC = {}".format(loglik, df, n_params, 2*(n_params-loglik), np.log(len(y))*n_params-2*loglik))
        varname = '$log_{10}$('+varname+')' if (use_log and varname is not None) else varname
        if plot_with_categorical:
            plot_lcga(gcm.pretty_beta(betas_opt), timesteps, y, degree, groups-min(groups), title='GCM w/ groups - degree {}'.format(degree), varname=varname)
        else:
            groups2plot = list(set([tuple(r.astype(int)) for r in groups_converted[:]]))
            extended_plot(betas_opt, timesteps, y, groups_converted, groups2plot, degree, title='GCM w/ groups - degree {}'.format(degree), varname=varname)
        print()

def run_pipeline_LCGA(y_main, timesteps, max_degree, min_degree=1, max_latent_classes=3, y_control=None, src_labels1D=None,
    R_struct='multiple_identity', varname=None, use_log=False):

    # 0- remove outliers and basic treatment
    # TODO

    # 1- basic assertions
    assert (
        (y_control is not None and src_labels1D is not None) or
        (y_control is None and src_labels1D is None)
    ), "in order to use harmonize data, both y_control and src_labels1D must be provided"
    assert min_degree>=1, "min_degree should be at least 1"
    assert not np.any(np.isnan(y_main)), "NaN (Not A Number) detected in y_main"
    assert not np.any(np.isinf(np.abs(y_main))), "Inf or -Inf detected in y_main"
    if y_control is not None:
        assert not np.any(np.isnan(y_control)), "NaN (Not A Number) detected in y_control"
        assert not np.any(np.isinf(np.abs(y_control))), "Inf or -Inf detected in y_control"

    # 2- log?
    if use_log:
        if np.any(y_main <= 0):
            print("We found negative values in y_main and the log transformation was NOT applied.\n" +
                "We maintained the original scale.\n") 
        elif y_control is not None and np.any(y_control <= 0):
            print("We found negative values in y_control and the log transformation was NOT applied.\n" +
                "We maintained the original scale.\n") 
        else:
            y_main = np.log10(y_main)
            y_control = np.log10(y_control) if y_control is not None else None
            print("We converted data to their logarithm (in base 10)")

    # 3 - LCGA
    print("==========================================================")
    print("      Latent Class Growth Analysis                        ")
    print("==========================================================\n")
    logliks = {} # loglikelihood of all models in the form {degree: {K: (loglik, n_params)}}
    for degree in range(max_degree, min_degree-1, -1):
        if degree >= len(timesteps)-1:
            print('degree {} ignored: We recommend to use only degrees lower than #timesteps - 1'.format(degree))
            continue
        logliks[degree] = {}
        # harmonize data:
        if y_control is not None:
            harmonizer = Harmonizer(y_main, y_control, src_labels1D, timesteps, degree)
            print("harmonizing data")
            try:
                _,_ = harmonizer.fit(verbose=False, force_solver=True)
            except AssertionError as err:
                print('something went wrong while harmonizing the data:')
                print(err)
                print("We will use the original data")
                y = np.copy(y_main)
                print()
            else:
                y,_ = harmonizer.transform(y_main, y_control, src_labels1D)
                print("data harmonized")
        else:
            y = np.copy(y_main)
        # apply model:
        for K in range(max_latent_classes,0,-1):
            print('degree {}, {} latent classes{}-\n'
                .format(degree, K, ' (simple regression)' if K==1 else '')
            )
            lcga = LCGA(y, timesteps, degree, K, R_struct)
            try:
                print('estimating latent classes...')
                Rs, betas, pis= lcga.solve(verbose=False)
            except Warning as war:
                print(war) # just print and go on
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
            print("Number of subjects assigned to each cluster (by greatest posterior probability):", [sum(preds == i) for i in range(K)])
            logliks[degree][K] = (lcga.get_loglikelihood(), lcga.get_n_params())
            varname = '$log_{10}$('+varname+')' if (use_log and varname is not None) else varname
            plot_lcga(betas, timesteps, y, degree, preds, title='LCGA - degree {} and {} clusters'.format(degree, K), varname=varname)
            print()
    # model comparison:
    if max_latent_classes > 2:
        print("generating graph of the log-likelihood for different combinations...")
        print("Note: Greater Ks and degrees always have a higher likelihood,"+
        " but are not necessarily better")
        plot_loglikelihoods(logliks, 'degree', 'K')
        print()
        print("generating graph of Akaike Information Criterion and Bayesian Information Criterion...")
        print("Slower values are supposed to indicate a better compromise between explaining data "+
        "and simplifying the model")
        print("For big datasets, BIC is usually preferred to AIC")
        print()
        plot_information_criterions(logliks, 'degree', 'K', len(y))
    # summary
    print('Summary: model selection')
    for k1 in logliks:
        for k2 in sorted(logliks[k1]):
            n_params = logliks[k1][k2][1]
            loglik = logliks[k1][k2][0]
            print('degree {}, K {} : loglik = {}, nb params = {}, AIC = {}, BIC = {}'
                .format(k1, k2, loglik, n_params, 2*(n_params-loglik), np.log(len(y))*n_params-2*loglik))        
    print()


