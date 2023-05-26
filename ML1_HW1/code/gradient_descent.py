import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_eggholder_function(f):
    '''
    Plotting the 3D surface of a given cost function f.
    :param f: The function to visualize
    :return:
    '''
    n = 1000
    bounds = [-512, 512]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    x_ax = np.linspace(bounds[0], bounds[1], n)
    y_ax = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(x_ax, y_ax)

    ZZ = np.zeros(XX.shape)
    ZZ = f([XX, YY])

    ax.plot_surface(XX, YY, ZZ, cmap='jet')
    plt.show()


def gradient_descent(f, df, x, learning_rate, max_iter):
    """
    Find the optimal solution of the function f(x) using gradient descent:
    Until the max number of iteration is reached, decrease the parameter x by the gradient times the learning_rate.
    The function should return the minimal argument x and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: function representing the gradient of f
    :param x: vector, initial point
    :param learning_rate:
    :param max_iter: maximum number of iterations
    :return: x (solution, vector), E_list (array of errors over iterations)
    """

    E_list = np.zeros(max_iter)
    print(df(x))
    E_list[0] = df(x) 
    for i in range(1, max_iter):
         x -= learning_rate * df(x)
         gradient = df(x)
         E_list[i] = gradient
    # Implement the gradient descent algorithm
    # E_list should be appended in each iteration, with the current value of the cost
    
    return x, E_list


def eggholder(x:list):
    """
    :param x: 2 variables within the list -> x coordinate, y coordinate
    :return: cost 
    """
    # Implement the cost function specified in the HW1 sheet
    # z = x[0] + x[1]  # TODO: change me
    x, y = x[0], x[1]
    a = (abs(x/2 + (y+47))) ** 0.5
    b = (abs(x - (y+47))) ** 0.5
    z = -(y+47) * np.sin(a) - x * np.sin(b)
    return z


def gradient_eggholder(x):
    """
    :param x: 2 variables within the list -> x coordinate, y coordinate
    """
    x, y = x[0], x[1]
    # Implement gradients of the Eggholder function w.r.t. x and y
    grad_x = 0  # TODO: change me 
    grad_y = 0  # TODO: change me
    grad_x = -np.sin((abs(x/2 + (y+47)))**2) - x/2 * (x/2 + (y+47))**-0.5 * (y+47) * np.cos((abs(x/2 + (y+47)))**0.5) - np.sin((abs(x - (y+47)))**0.5) - np.sin((abs(x - (y+47)))**0.5) * (x - (y+47))**-0.5
    grad_y = -np.sin((abs(x/2 + (y+47)))**0.5) * (y+47) * (1/2 * (x/2 + (y+47))**-0.5) - (x * np.cos((abs(x - (y+47)))**0.5) * (1/2 * (x - (y+47)))**-0.5 * (x - (y+47))) - np.sin((abs(x/2 + (y+47)))**0.5) * (1/2 * (x/2 + (y+47)))**-0.5

                                      
    return np.array([grad_x, grad_y])


def generic_GD_solver(x):
    # TODO: bonus task
    return 0
