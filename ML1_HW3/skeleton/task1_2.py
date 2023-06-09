import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

import plotting
from datasets import get_toy_dataset
from task1_1 import KNearestNeighborsClassifier

if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    knn = KNearestNeighborsClassifier()
    
    parameters = {'k': [1, 5, 20, 50, 100]}
    # print(parameters['k'])

    #TODO: use the `GridSearchCV` meta-classifier and search over different values of `k`!
    # include the `return_train_score=True` option to get the training accuracies
    clf = GridSearchCV(knn, parameters, return_train_score=True)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Test Score: {test_score}")
    print(f"Dataset {idx}: {clf.best_params_}")

    plt.figure()
    plotting.plot_decision_boundary(X_train, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    plt.title(f'Dataset {idx}')
    # TODO you should use the plt.savefig(...) function to store your plots before calling plt.show()
    plt.savefig(f'../plots/decision_boundary_dataset_{idx}.jpg', dpi=120)
    plt.show()
    plt.close()
    plt.figure()
    plt.plot(clf.cv_results_['mean_train_score'],  label="train")
    plt.plot(clf.cv_results_['mean_test_score'], label="test")
    # plt.xlabel('Values of K')
    plt.ylabel('Mean Score')
    plt.title(f'Dataset {idx}')
    plt.legend()
    plt.savefig(f'../plots/dataset_{idx}.jpg', dpi=120)
    plt.show()
    # plt.close()

