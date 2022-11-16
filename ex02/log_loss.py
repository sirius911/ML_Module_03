import math
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
        # print(x)
        x_ = np.hstack((np.ones((m, 1)), x))
        # print(x_)
        return sigmoid_(x_ @ theta)

    except Exception as e:
        print(e)
        return None

def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        print("Error in log_loss_() : not numpy array.")
        return None
    if not isinstance(eps, float):
        print("Error in log_loss_() : eps must be a float.")
        return None
    try:
        m = y.shape[0]
        if y.shape != y_hat.shape or y.shape[1] != 1:
            print("Error in log_loss_() : incompatible shape.")
            return None

        y_hat = y_hat + eps
        inter = []
        for i in range(m):
            elem = (y[i] * math.log(y_hat[i])) + ((1 - y[i]) * math.log(1 - y_hat[i]))
            inter.append(elem)
        return float(-(1 / m) * sum(inter))

    except Exception as e:
        print(e)
        return None