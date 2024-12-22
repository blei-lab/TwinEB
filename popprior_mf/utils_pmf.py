import numpy as np

def fast_digamma(x):
    """Compute fast digamma function
    
    It assumes x > 0.

    Written by Tim Vieira and based on the lightspeed package by Tom Minka
    source: github.com/timvieira/Notes-on-scipy.special.digamma.ipynb
    """
    r = 0
    while (x<=5):
        r -= 1/x
        x += 1
    f = 1/(x*x)
    t = f*(-1/12.0 + f*(1/120.0 + f*(-1/252.0 + f*(1/240.0 + f*(-1/132.0
        + f*(691/32760.0 + f*(-1/12.0 + f*3617/8160.0)))))))
    return r + np.log(x) - 0.5/x + t


def sum_sparse_matrix(A, axis):
    """Compute sums along axes of sparse matrix in coordinate list format
    
    Args:
        A: np.array with three columns. First column is row indices, 
            second column is column indices, and third column is values.
        axis: 0 for summing along columns, 1 for summing along rows.
    
    Returns:
        A list of sums along the specified axis. The length of the list is 
        equal to the number of unique row or column indices.
    """
    axis = abs(1 - axis)
    sums = np.bincount(A[:, axis], weights = A[:, 2])
    return sums[sum!=0]


def extract_users(interactions):
    """Extract dictionary with list of interactions for each user
    
    Args:
        interactions: np.array with three columns. First column is row indices, 
            second column is column indices, and third column is values.
    
    Returns:
        A dictionary with: keys = users; values = lists of row indices
            corresponding to users interactions. Each list is a np.array.
    """
    users = {}
    for i in range(interactions.shape[0]):
        user = interactions[i, 0]
        if user not in users:
            users[user] = np.array([i])
        else:
            users[user] = np.append(users[user], i)
    return users