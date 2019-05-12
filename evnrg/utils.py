import numba as nb
import numpy as np

@nb.njit(cache=True)
def isin_(x, arr):
    """
    This function only exists because Numba (stupidly?) doesn't support 
    the 'in' keyword or np.isin().
    """
    exists = False
    for i in range(arr.shape[0]):
        if arr[i] == x:
            exists = True
            break
    return exists