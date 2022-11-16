import numpy as np


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        print("Error in sigmoid(): x is not a numpy array.")
        return None
    if x.size == 0:
        print("Error in sigmoid(): Empty array")
        return None
    try:
        m = x.shape[0]
        n = x.shape[1]
        if n != 1:
            print(f"Error sigmoid(): X.shape = {x.shape} != (m, 1).")
            return None
        ret = np.zeros(x.shape)
        ret = 1 / (1 + np.exp(-x))
        return ret
    except Exception as e:
        print(e)
        return None