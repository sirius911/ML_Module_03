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

def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        print("Error logistic_predict_(): not numpy arrays.")
        return None
    if x.size == 0 or theta.size == 0:
        print("Error logistic_predict_(): empty array.")
        return None
    try:
        m = x.shape[0]
        n = x.shape[1]
        if theta.shape[0] != n + 1 or theta.shape[1] != 1:
            print("Error logistic_predict_(): Incompatible shape.")
            return None
        x_ = np.hstack((np.ones((m, 1)), x))
        return sigmoid_(x_ @ theta)

    except Exception as e:
        print(e)
        return None

def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatibl
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of shape (n+1) * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        print("Error in log_gradient(): x is not a numpy array.")
        return None
    if x.size == 0 or y.size == 0:
        print("Error in log_gradient(): Empty array")
        return None
    try:
        m, n =x.shape
        if m != y.shape[0] or y.shape[1] != 1 or theta.shape[0] != n + 1:
            print("Error in log_gradient(): Empty array")
            return None
        y_hat = logistic_predict_(x, theta)
        gradient = np.zeros((n + 1, 1))
        gradient[0] = (1 / m) * sum(y_hat -y)
        for j in range(1, n + 1):
            sum_ = 0
            for i in range(m):
                sum_ += (y_hat[i] - y[i]) * x[i][j - 1]
                gradient[j] = (1 / m) * sum_
        return gradient
    except Exception as e:
        print(f"Error in log_gradient(): {e}")
        return None