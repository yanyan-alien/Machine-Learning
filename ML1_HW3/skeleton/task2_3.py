import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import plotting
from datasets import get_toy_dataset

if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    svc = SVC(tol=1e-4)
    # TODO perform grid search, decide on suitable parameter ranges and state sensible parameter ranges in your report
    params = {'kernel':['linear', 'rbf'], 'gamma': ['scale', 'auto'], 'C': [0.0005, 0.005, 0.1, 0.5, 1.0]}
    clf = GridSearchCV(svc, param_grid=params)
    clf.fit(X_train, y_train)
    cv_test_score = clf.score(X_test, y_test)
    print("Mean Cross-Validated test Score:", cv_test_score)
    best_svc = clf.best_estimator_
    print("Test Score:", best_svc.score(X_test, y_test))
    print(f"Dataset {idx}: {clf.best_params_}")

    # TODO plot and save decision boundaries
    plt.figure()
    plotting.plot_decision_boundary(X_train, best_svc)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    plt.title(f"Dataset {idx}")
    plt.savefig(f'../plots/task2_3_decision_boundary_dataset_{idx}.jpg')
    plt.show()
