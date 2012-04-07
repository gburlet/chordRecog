import numpy as np

def logsumexp(lnX, axis=0):
    '''
    Calculates the sum of the matrix lnX along the specified axis
    assuming X is in the log domain. The returned sum is still in
    the log domain. Minimizes possibility of underflow/overflow
    by normalizing with respect to the max value along the summation.
    (derived from scikit-learn)
    
    PARAMETERS
    ----------
    lnX {NxM}: matrix in the log domain
    axis: axis to sum over (0 = rows, 1 = columns)

    RETURNS
    -------
    lnXsum (N,) or (M,): log(sum(exp(lnX)))
    '''

    lnXmax = np.max(lnX, axis=axis)
    
    # rollaxis returns a copy, so original lnX isn't modified
    lnX = np.rollaxis(lnX, axis=axis)

    lnX_norm = lnX - lnXmax
    # clamp to avoid overflow/underflow
    lnX_norm[lnX_norm <= -150] = -150
    lnX_norm[lnX_norm >= 150] = 150
    lnXsum = np.log(np.sum(np.exp(lnX_norm), axis=0)) + lnXmax

    return lnXsum

def unsqueeze(X, nDimDesired, axis=0):
    '''
    Helper function. Only expands if the the current number of dimensions
    is less than the desired number of dimensions.
    '''
    if X.ndim < nDimDesired:
        return np.expand_dims(X, axis=axis)
    else:
        return X
