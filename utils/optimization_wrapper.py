import numpy as np
import scipy.optimize as optimize

def gcm_FIML_minimizer(obj, shapes=None, init_func=None, max_y=1, min_y=0, method='BFGS', multistart=False, verbose=True):
    """optimization wrapper for GCM models that calls gcm_minimizer in a slightly simpler manner

    Args:
        obj (callable): objective function
        shapes (list, optional): list of shapes of beta (int), D (int or tuple), and R (int or tuple).
                                 In the case of a matrix (i.e. R or D) of size n, the value should be
                                 n, (n,), or (n,n). If None, init_func should be provided.
                                 Defaults to None.
        init_func (callable, optional): Function with no arguments that initialize the starting
                                        point for the  optimization. If None, the argument shapes
                                        should be provided. Defaults to None.
        max_y (int, optional): A 'reasonable' upper-bound for beta[0]. Defaults to 1.
        min_y (int, optional): A 'reasonable' lower-bound for beta[0]. Defaults to 0.
        method (str, optional): Optimization method as called by Scipy optimization.minimize.
                                Only 'BFGS' and 'TNC' accepted. Defaults to 'BFGS'.
        multistart (bool, optional): Whether to use multistart (20 repetitions) or single-start.
                                     Defaults to False (single-start).
        verbose (bool, optional): Verbose mode or not. Defaults to True.

    Raises:
        ValueError: if the shape of D or R is not correct (for example a shape (n,m) with m != n)

    Returns:
        1D array, boolean: respectively, the optimum theta (the best one in case of multistart)
                           and a boolean indicating optimization success (true if at least half
                           of the tests converged, in the case of multistart, or if the only test
                           converged, in the case of single-start)
    """
    return gcm_minimizer(obj, shapes, init_func, True, max_y, min_y, method, multistart, verbose)

def gcm_minimizer(obj, shapes=None, init_func=None, init_for_lavaan=False, max_y=1, min_y=0, method='BFGS',
 multistart=False, verbose=True):
    """optimization wrapper for GCM models (takes care of initialization, multistart, and errors)

    Args:
        obj (callable): objective function
        shapes (list, optional): list of shapes of beta (int), D (int or tuple), and R (int or tuple).
                                 In the case of a matrix (i.e. R or D) of size n, the value should be
                                 n, (n,), or (n,n). If None, init_func should be provided.
                                 Defaults to None.
        init_func (callable, optional): Function with no arguments that initialize the starting
                                        point for the  optimization. If None, the argument shapes
                                        should be provided. Defaults to None.
        init_for_lavaan (bool, optional): This is the initialization method for R and D.
                                          If True, we initialize R and D so that they are
                                          semi-definite positive when interpreted by the
                                          objective function that folows lavaan representation
                                          of covariance matrices. Defaults to False.
        max_y (int, optional): A 'reasonable' upper-bound for beta[0]. Defaults to 1.
        min_y (int, optional): A 'reasonable' lower-bound for beta[0]. Defaults to 0.
        method (str, optional): Optimization method as called by Scipy optimization.minimize.
                                Only 'BFGS' and 'TNC' accepted. Defaults to 'BFGS'.
        multistart (bool, optional): Whether to use multistart (20 repetitions) or single-start.
                                     Defaults to False (single-start).
        verbose (bool, optional): Verbose mode or not. Defaults to True.

    Raises:
        ValueError: if the shape of D or R is not correct (for example a shape (n,m) with m != n)

    Returns:
        1D array, boolean: respectively, the optimum theta (the best one in case of multistart)
                           and a boolean indicating optimization success (true if at least half
                           of the tests converged, in the case of multistart, or if the only test
                           converged, in the case of single-start)
    """

    assert init_func or shapes
    assert init_func is None or shapes is None
    if shapes:
        assert isinstance(shapes, tuple) or isinstance(shapes, list)
        assert isinstance(shapes[0],int) # beta
        assert isinstance(shapes[1],int) or isinstance(shapes[1],tuple) # R
        assert isinstance(shapes[2],int) or isinstance(shapes[2],tuple) # D

    assert method in ['BFGS','TNC'], 'Please chose a valid optimization method'

    theta_opt = None
    opt_val = np.Inf
    opt_val_list = []
    nb_successes = 0
    opt_message = None

    n_iter = 20 if multistart else 1

    for _ in range(n_iter):

        ### Initialization ###
        if init_func:
            theta_0 = init_func()
        else:
            p, shape_R, shape_D = shapes
            delta = max_y-min_y # scale in initialization for beta[0]
            offset = (max_y+min_y)/2 - delta/2 # translation for initialization for beta[0]
            eps = 1E-6
            #
            beta_0 = np.random.rand(p) if multistart else np.zeros(p)
            beta_0[0] = beta_0[0] * delta + offset
            #
            if isinstance(shape_R, int):
                nR = shape_R
                R_0 = np.abs(np.random.rand(nR)) + eps*np.ones(nR) # strictly positive array
            elif isinstance(shape_R, tuple) and len(shape_R) == 1:
                nR = shape_R[0]
                R_0 = np.abs(np.random.rand(nR)) + eps*np.ones(nR) # strictly positive array
            elif not init_for_lavaan:
                assert len(shape_R) == 2 and shape_R[0] == shape_R[1]
                nR = shape_R[0]*(shape_R[0]+1)//2
                R_0 = np.random.rand(nR)
            elif init_for_lavaan:
                assert len(shape_R) == 2 and shape_R[0] == shape_R[1]
                T = shape_R[0]
                nR = shape_R[0]*(shape_R[0]+1)//2
                temp = np.random.rand(T, T) + eps*np.ones((T, T))
                R_0 = (temp.T @ temp)[np.triu_indices(T)].flatten() # make R_0 definite-positive
            else:
                raise ValueError
            #
            if isinstance(shape_D, int):
                nD = shape_D
                D_0 = np.abs(np.random.rand(nD)) + eps*np.ones(nD) # strictly positive array
            elif isinstance(shape_D, tuple) and len(shape_D) == 1:
                nD = shape_D[0]
                D_0 = np.abs(np.random.rand(nD)) + eps*np.ones(nD) # strictly positive array
            elif not init_for_lavaan:
                assert len(shape_D) == 2 and shape_D[0] == shape_D[1]
                nD = shape_D[0]*(shape_D[0]+1)//2
                D_0 = np.random.rand(nD)
            elif init_for_lavaan:
                assert len(shape_D) == 2 and shape_D[0] == shape_D[1]
                k = shape_D[0]
                nD = shape_D[0]*(shape_D[0]+1)//2
                temp = np.random.rand(k, k) + eps*np.ones((k, k))
                D_0 = (temp.T @ temp)[np.triu_indices(k)].flatten() # make D_0 definite-positive
            else:
                raise ValueError
            #
            theta_0 = np.zeros(p+nR+nD)
            theta_0[0:p] = beta_0
            theta_0[p:p+nR] = R_0
            theta_0[p+nR:p+nR+nD] = D_0

        ### Optimization ###
        optimize_res = optimize.minimize(obj, theta_0, jac='3-point', method=method,
        options={'maxiter':1000, 'disp':verbose if not multistart else False})
        theta = optimize_res.x
        val = obj(theta)
        if val < opt_val:
            opt_val = val
            theta_opt = theta
            opt_message = optimize_res.message
        opt_val_list.append(val)
        if optimize_res.success:
            nb_successes += 1
        
    if verbose and multistart:
        print('{}/{} succeded optimizations'.format(nb_successes, n_iter))
        print('The biggest and smallest (best) objective function evaluations at the end of some run were:')
        print(max(opt_val_list))
        print(opt_val)
        print('Message of the best run: {}'.format(opt_message))

    return theta_opt, nb_successes == 1 if not multistart else nb_successes >= 10



        




