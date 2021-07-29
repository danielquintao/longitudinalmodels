import numpy as np

def convert_label(groups, offset=0):
    """converts categorical labels (eg. 1,2,2,1,2,1) to special one-hot notation required by the solver
     Args:
        groups (1D array or array of shape (N_classes,1)): Numpy array with the categorical labels. The labels
                                                           should be of type int, and should vary from 0 to N_classes-1
                                                           or from 1 to N_classes, with at least one representant
                                                           per group
        offset (1 or 0, optional): The smallest groups label. Defaults to 0.
     Returns:
        2D array: special one-hot encoding of the labels, where the smallest label becomes (0,0,..,0)
                  N_classes-1 times, and the other labels are (1,0,..0), (0,1,..,0), ..., (0,0,..,1)
    """
    assert groups.shape == (len(groups),) or groups.shape == (len(groups),1), "wrong format"
    groups = groups.flatten() if len(groups.shape) == 2 else groups
    assert np.issubdtype(groups.dtype, np.integer), 'group labels should be integers'
    assert offset == 0 or offset == 1, "the slowest label should be 0. Check your convention please"
    max_element = max(groups)
    for g in range(offset, max_element+1):
        assert np.any(groups == g), "every group should have a representant"
    assert all([g in groups for g in np.arange(offset, max_element+1)]), "strange label"
    assert min(groups) == offset, "the minimum eleent is not equal to the offset... Please check your args"
    N_groups = max_element - offset + 1
    groups_ = np.zeros((len(groups), N_groups-1)) # -1 because class offset is represented by null vector
    for g in range(1,N_groups): # we start from 1 because class offset is represented by null vector
        groups_[groups==g+offset,g-1] = 1
    return groups_
