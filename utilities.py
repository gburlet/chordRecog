import numpy as np

def logsumexp(lnX, axis=0):
    '''
    Calculates the sum of the matrix lnX along the specified axis
    assuming X is in the log domain. The returned sum is still in
    the log domain. Minimizes possibility of underflow/overflow
    by normalizing with respect to the max value along the summation.
    (from scikit-learn)
    
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

    lnXsum = np.log(np.sum(np.exp(lnX - lnXmax), axis=0)) + lnXmax

    # for floating point errors ... there's nothing else we can do here
    # replace NaN or inf with the max lnP
    # taken from Matlab GMM EM library: 
    # http://www.mathworks.com/matlabcentral/fileexchange/26184-em-algorithm-for-gaussian-mixture-model
    errInd = ~np.isfinite(lnXsum)

    if lnXsum.ndim > 1:
        lnXsum[errInd] = lnXmax[errInd]
    elif errInd:
        # for arrays where the sum comes out to a single float
        lnXsum = lnXmax

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
