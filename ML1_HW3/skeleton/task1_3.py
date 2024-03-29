import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

import plotting
from datasets import get_toy_dataset
from task1_1 import KNearestNeighborsClassifier
from sklearn.model_selection import cross_val_score



if __name__ == '__main__':
  X_train, X_test, y_train, y_test = get_toy_dataset(2, apply_noise=True)
  for k in [1, 5, 20, 50, 100]:
    # TODO fit your KNearestNeighborsClassifier with k in {1, 5, 20, 50, 100} and plot the decision boundaries
    clf = KNearestNeighborsClassifier(k=k)
    clf.fit(X_train, y_train)
    # TODO you can use the `cross_val_score` method to manually perform cross-validation
    cross_score = cross_val_score(clf, X_train, y_train)
    print(f"Cross Validated Score for k={k}: {cross_score}")
    # TODO report mean cross-validated scores!
    test_score = clf.score(X_test, y_test)
    print(f"Test Score for k={k}: {test_score}")
    # TODO plot the decision boundaries!

    plt.figure()
    plotting.plot_decision_boundary(X_train, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    plt.title(f'k = {k}')
    plt.savefig(f'../plots/decision_boundary_task1_3_k_{k}.jpg', dpi=120)
    plt.show()
    plt.close()
    
  # TODO find the best parameters for the noisy dataset!
  knn = KNearestNeighborsClassifier()
  parameters = {'k': [1, 5, 20, 50, 100]}
  clf = GridSearchCV(knn, parameters, return_train_score=True)
  clf.fit(X_train, y_train)
  # TODO The `cv_results_` attribute of `GridSearchCV` contains useful aggregate information
  # such as the `mean_train_score` and `mean_test_score`. Plot these values as a function of `k` and report the best
  # parameters. Is the classifier very sensitive to the choice of k?
  plt.figure()
  plt.plot(clf.cv_results_['mean_train_score'], label="train")
  plt.plot(clf.cv_results_['mean_test_score'], label="test")
  plt.savefig(f'../plots/training_validation_accuracy_vary_k_task1_3.jpg', dpi=120)
  plt.legend()
  plt.show()
