from GCM_basic import DiagGCMSolver, TimeIndepErrorGCMSolver
from GCM_basic import DiagGCMLavaanLikeSolver, TimeIndepErrorGCMLavaanLikeSolver
from GCM_extended import DiagExtendedGCMSolver, TimeIndepErrorExtendedGCMSolver
from GCM_extended import DiagExtendedGCMLavaanLikeSolver, TimeIndepErrorExtendedGCMLavaanLikeSolver

# factory function
def GCM(y, timesteps, degree, R_struct='multiple_identity', groups=None, lavaan_like=False):
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

