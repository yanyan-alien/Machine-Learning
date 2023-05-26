import numpy as np
from numpy.linalg import pinv  # if you decide to implement the equations using the matrix notation


def test_fit_zero_intercept_lin_model():
    # TODO: bonus task
    # Implement two test cases that test your implementation of the function fit_zero_intercept_lin_model. Use assert command for that.
    return 0


def test_fit_lin_model_with_intercept():
    # TODO: bonus task
    # Implement two test cases that test your implementation of the function fit_lin_model_with_intercept. Use assert command for that.
    return 0


def fit_zero_intercept_lin_model(x, y):
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta 
    """
    theta = sum(y) / sum(x)
    return theta


def fit_lin_model_with_intercept(x, y):
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta_0, theta_1 
    """
    x = x[None].T    
    n_data_points = len(x)
    z = np.hstack((np.ones((n_data_points, 1)), x))
    theta = pinv(z).dot(y)
    theta_0, theta_1 = theta[0], theta[1]  
    return theta_0, theta_1 


