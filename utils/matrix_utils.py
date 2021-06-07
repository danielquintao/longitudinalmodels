import numpy as np

def triangular2flattened(x):
    """Receives a square matrix and returns a flat array with upper-diagonal elements

    Args:
        x (ndarray): ndarray of shape (n,n), ideally upper-diagonal or symmetric

    Returns:
        ndarray: flattened ndarray of length n*(n+1)/2 with elements x11,x12,...,x1n,x22,...,xnn of x
    """
    assert len(x.shape) == 2
    assert x.shape[0] == x.shape[1]
    return x[np.triu_indices(x.shape[0])]

def flattened2triangular(flat, n=None):
    """Receives a flattened list of size n*(n+1)/2 and returns upper-triangular square matrix

    Args:
        flat (ndarray): ndarray with values [a11,a12,...,a1n,a22,...,ann]
        n [optional] (int): dimention of the triangular matrix, such that n*(n+1)/2 = len(flat)

    Returns:
        ndarray: triangular matrix of shape (n,n) with upper-diagonal valuer filled with elements of flat
    """
    if n is None:
        N = len(flat)
        # solves basic equation n*(n-1)/2 = N
        n = (-1. + np.sqrt(1.+8.*N))/2
    else:
        assert len(flat) == n*(n+1)/2
    assert float(n).is_integer()
    n = int(n)
    sqr = np.zeros((n,n))
    sqr[np.triu_indices(n)] = flat
    return sqr

## Test
# a = np.array([1,2,3,4,5,6])
# sqr = flattened2triangular(a)
# print(sqr)
# print(triangular2flattened(sqr))