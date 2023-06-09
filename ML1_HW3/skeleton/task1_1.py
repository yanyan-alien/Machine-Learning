import numpy as np
from sklearn.base import BaseEstimator


class KNearestNeighborsClassifier(BaseEstimator):
  def __init__(self, k=1):
    self.k = k

  def fit(self, X, y):
    '''
    :param X: data matrix X [based on other classifier]
    :param Y: target(s) Y [based on other classifier]

    :return: self: object (trained model) [based on other classifier] 
    '''
    # TODO IMPLEMENT ME
    # store X and y
    self.X = X
    self.y = y
    return self

  def score(self, X, y):
    y_pred = self.predict(X)
    return np.mean(y_pred == y)

  def predict(self, X):
    '''
    :param X: X_test; not restricted to just 1 point 
    '''
    # TODO: assign class labels
    # useful numpy methods: np.argsort, np.unique, np.argmax, np.count_nonzero
    # pay close attention to the `axis` parameter of these methods
    # broadcasting is really useful for this task!
    # See https://numpy.org/doc/stable/user/basics.broadcasting.html
    
    k = self.k
    point = X
    data = self.X
    targets = []
    for point in X:
      distances = np.sum((data - point)**2, axis=1) ** 0.5 
      new_arr = np.column_stack((distances, self.y))
      ixs = np.argsort(new_arr[:,0])
      ixs = np.column_stack((ixs, ixs))
      sorted_arr = np.take_along_axis(new_arr, ixs, axis=0)

      k_arr_y = sorted_arr[:k,1]
      values, counts = np.unique(k_arr_y, return_counts=True)
      ind = np.argmax(counts)
      targets.append(values[ind])

    np_targets = np.array(targets)
    return np_targets
  
    # https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-array
    # https://stackoverflow.com/questions/62816422/how-do-i-sort-by-first-item-of-array-element
    # https://stackoverflow.com/questions/17710672/create-a-two-dimensional-array-with-two-one-dimensional-arrays
