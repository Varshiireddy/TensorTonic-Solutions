import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    arr = np.asarray(x, dtype=float)
    result = 1 / (1 + np.exp(-arr))

    if np.isscalar(x):        # input oka single number aa?
        return float(result)  # aithe plain float ivvu
    return result   