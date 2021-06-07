from GCM_basic import DiagGCMSolver, TimeIndepErrorGCMSolver
from GCM_basic import DiagGCMLavaanLikeSolver, TimeIndepErrorGCMLavaanLikeSolver
from GCM_extended import DiagExtendedGCMSolver, TimeIndepErrorExtendedGCMSolver
from GCM_extended import DiagExtendedGCMLavaanLikeSolver, TimeIndepErrorExtendedGCMLavaanLikeSolver

# factory function
def GCM(y, timesteps, degree, R_struct='multiple_identity', groups=None, lavaan_like=False):
    """returns GCM-solver object

    Args:
        y (2D array): observations : each row an individual, each column a time step
        timesteps (array): time steps, e.g. np.array([0,1,2,3])
        degree (int): degree of the polynomial to fit
        R_struct (str, optional): Structure to use for the errors covariance matrix.
                                  Possible values: 'multiple_identity', 'diagonal'.
                                  Defaults to 'multiple_identity'.
        groups (binary ndarray, optional): membership of individuals in each group (0 or 1), None if
                                           there are no groups. Defaults to None.
                                           Note 1: the groups are supposed to share the random effects cov matrix
                                           Note 2: we can use (0,0,..,0) to represent a group, or start from (1,0,..,0)
                                                   In either case, the estimator method of the returned object returns the
                                                   "neutral" fixed effects for (0,0..,0) before (1,0,..,0) etc
        lavaan_like (bool, optional): Whether to enfore covriance matrices to be semi-definite positive in representation
                                      (using squared numbers and Cholesky decomposition) or not. Defaults to False.

    Raises:
        ValueError: invalid argument R_struct

    Returns:
        [no fixed type]: GCM Solver object
    """

    if R_struct == 'multiple_identity':
        if groups is None:
            if lavaan_like:
                return TimeIndepErrorGCMLavaanLikeSolver(y, timesteps, degree)
            else:
                return TimeIndepErrorGCMSolver(y, timesteps, degree)
        else:
            if lavaan_like:
                 return TimeIndepErrorExtendedGCMLavaanLikeSolver(y, groups, timesteps, degree)
            else:
                return TimeIndepErrorExtendedGCMSolver(y, groups, timesteps, degree)
    elif R_struct == 'diagonal':
        if groups is None:
            if lavaan_like:
                return DiagGCMLavaanLikeSolver(y, timesteps, degree)
            else:
                return DiagGCMSolver(y, timesteps, degree)
        else:
            if lavaan_like:
                return DiagExtendedGCMLavaanLikeSolver(y, groups, timesteps, degree)
            else:
                return DiagExtendedGCMSolver(y, groups, timesteps, degree)
    else:
        raise ValueError('possible values for R_struct are \'multiple_identity\' and \'diagonal\'')

