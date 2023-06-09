import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator

import plotting
from datasets import get_toy_dataset


def loss(w, b, C, X, y):
  # TODO: implement the loss function (eq. 1)
  # useful methods: np.sum, np.clip
  max_compare = 1- y * ((X @ w) + b)
  results = w @ w * 0.5 + C * np.sum(np.where(max_compare > 0, max_compare, 0))
  return results
  return np.inf


def grad(w, b, C, X, y):
  # TODO: implement the gradients with respect to w and b.
  # useful methods: np.sum, np.where, numpy broadcasting
  max_part = 1 - y * ((X @ w) + b)

  # # print(max_part.shape)

  max_b_part = np.where( max_part > 0, y, 0)
  
  w_y = np.column_stack((y, y)) 
  w_max_choice = np.where( max_part > 0, 1, 0)
  # print('max w part')
  max_w_part = w_y * X * np.column_stack((w_max_choice, w_max_choice))
  
  grad_w = w - C * np.sum(max_w_part)
  grad_b = - C * np.sum(max_b_part)
  return grad_w, grad_b


class LinearSVM(BaseEstimator):

  def __init__(self, C=1, eta=1e-3, max_iter=1000):
    self.C = C
    self.max_iter = max_iter
    self.eta = eta

  def fit(self, X, y):
    # TODO: initialize w and b. Does the initialization matter?
    # convert y: {0,1} -> -1, 1
    y = np.where(y == 0, -1, 1)
    np.random.seed(0)
    self.w = np.random.normal(size=X.shape[1])
    self.b = 0.
    loss_list = []

    for j in range(self.max_iter):
      # TODO: compute the gradients, update the weights, compute the loss
      grad_w, grad_b = grad(self.w, self.b, self.C, X, y)
      # print(grad_w, grad_b)

      # self.w = ...
      # self.b = ...
      self.w -= self.eta * grad_w
      self.b -= self.eta * grad_b
      loss_list.append(loss(self.w, self.b, self.C, X, y))

    return loss_list

  def predict(self, X):
    # TODO: assign class labels to unseen data

    hyperplane = (X @ self.w + self.b)
    # print(hyperplane)
    y_pred = np.where(hyperplane >= 0, 1, 0)
    # converting y_pred from {-1, 1} to {0, 1}
    return y_pred
    # return np.where(y_pred == -1, 0, 1)

  def score(self, X, y):
    y_pred = self.predict(X)
    return np.mean(y_pred == y)
