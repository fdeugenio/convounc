import numpy as np

from scipy import convolve


def convounc1d(a, v, cov=None, mode='full'):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.
    If an optional covariance matrix is passed, it returns the covariance
    matrix of the convolution.

    A wrapper to `scipy.convolve` (see relevant documentation). Notice that
    `convounc1d` is not symmetric (unlike the standard convolution operation),
    because it is assumed that `v` has 0 covariance.
    
    The convolution operator is often seen in signal processing, where it
    models the effect of a linear time-invariant system on a signal [1]_.  In
    probability theory, the sum of two independent random variables is
    distributed according to the convolution of their individual
    distributions.
    
    Parameters
    ----------
    a : (N,) array_like
        First one-dimensional input array. N >= M.
    v : (M,) array_like
        Second one-dimensional input array. M <= N.
    mode : {'full', 'valid', 'same'}, optional
        'full':
          By default, mode is 'full'.  This returns the convolution
          at each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.
    
        'same':
          Mode 'same' returns output of length ``max(M, N)``.  Boundary
          effects are still visible.
    
        'valid':
          Mode 'valid' returns output of length
          ``max(M, N) - min(M, N) + 1``.  The convolution product is only given
          for points where the signals overlap completely.  Values outside
          the signal boundary have no effect.
    cov : (N,), (N, N), array_like, optional
        the NxN covariance matrix for `a`. If `cov` is diagonal, it can be given
        as a (N,) array.
    
    Returns
    -------
    out_arr : ndarray
        Discrete, linear convolution of `a` and `v`.
    out_cov : ndarray, optional
        Covariance matrix of the discrete, linear convolution of `a` and `v`.
    
    See Also
    --------
    scipy.convolve: Convolve two arrays.
    """

    assert a.ndim == v.ndim == 1, '`a` and `v` must be 1-d arrays'

    assert a.size >= v.size, '`a` must be greater than `v`.'

    out_arr = convolve(a, v, mode=mode)
    out_size = out_arr.size

    if cov is None:
        return out_arr
    else:
        assert (
            ((cov.ndim == a.ndim) and (cov.size==a.size))
            or np.all(cov.shape == (a.size, a.size))), \
            '`cov` must be a 1-d array of (uncorrelated) variances, or a ' \
            '2-d covariance matrix appropriate for `a`.'

        # Instantiate cov as a diagonal matrix, if required.
        cov = np.diag(cov) if cov.ndim==1 else cov

        M, Mo2 = v.size, v.size // 2
        temp_size = (a.size + v.size + 1)
        J = np.zeros((temp_size, temp_size))

        skip = (temp_size - a.size) // 2

        temp = np.zeros(temp_size)
        temp[skip:-skip] = a
        a = temp
        temp_c = np.zeros((temp_size, temp_size))
        temp_c[skip:-skip, skip:-skip] = cov
        cov = temp_c

        for i in range(temp_size):
            for j in range(temp_size):
                if i-j+Mo2 >=0 and i-j+Mo2<=M-1:
                    J[i, j] = v[i-j+Mo2]

        out_cov = np.einsum('il,lm,jm->ij', J, cov, J)

        """ Only useful for diagonal covariance.
        Jd = np.zeros((temp_size, v.size))
        for i in range(temp_size):
            for j in range(v.size):
                true_j = j + i - 1
                if ((true_j>=0) and (true_j<temp_size)
                    and (i-true_j+Mo2 >=0) and (i-true_j+Mo2<=M-1)):
                    Jd[i, j] = v[i-true_j+Mo2]
        """

        skip = (temp_size - out_size) // 2

        out_cov = out_cov[skip:-skip, skip:-skip]

        return out_arr, out_cov
