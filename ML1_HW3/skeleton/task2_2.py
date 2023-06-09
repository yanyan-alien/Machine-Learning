import matplotlib.pyplot as plt

import plotting
from datasets import get_toy_dataset
from task2_1 import LinearSVM
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
  dataset = 1
  remove = True if dataset ==  1 else False
  X_train, X_test, y_train, y_test = get_toy_dataset(dataset, remove_outlier=remove)
  svm = LinearSVM()
  # TODO use grid search to find suitable parameters!
  params = {'C': [0.0005, 0.005, 0.1, 0.5, 1.0], 'max_iter': [1000, 2000, 5000, 10000], 'eta': [1e-3, 1e-4, 1e-5]}
  clf = GridSearchCV(svm, params)
  clf.fit(X_train, y_train)

  print("best parameters:",clf.best_params_)
  print("cross validated score:",clf.best_score_)
  # print('loss list', clf.best_estimator_.fit(X_train, y_train)[-1])
  # print(svm.predict(X_test))

  # TODO Use the parameters you have found to instantiate a LinearSVM.
  # the `fit` method returns a list of scores that you should plot in order
  # to monitor the convergence. When does the classifier converge?
  
  # svm = LinearSVM(...)
  svm = clf.best_estimator_
  scores = clf.best_estimator_.fit(X_train, y_train)
  plt.figure()
  plt.plot(scores)
  plt.title('Convergence Graph')
  plt.savefig(f'../plots/task2_2_convergence_graph_dataset_{dataset}_.jpg')
  # plt.show()
  plt.close()
  
  test_score = clf.score(X_test, y_test)
  print(f"Test Score: {test_score}")
  svm_test_score = clf.best_estimator_.score(X_test, y_test)
  print(f"svm Test Score: {svm_test_score}")

  
  # TODO plot the decision boundary!
  plt.figure()
  plotting.plot_decision_boundary(X_train, clf.best_estimator_)
  plotting.plot_dataset(X_train, X_test, y_train, y_test)
  plt.savefig(f'../plots/task2_2_decision_boundary_dataset_{dataset}.jpg')
  plt.show()
  # plt.close()
