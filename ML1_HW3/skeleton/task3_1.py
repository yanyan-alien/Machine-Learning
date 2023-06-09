import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import plotting
from datasets import get_toy_dataset

if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    # TODO start with `n_estimators = 1`
    # start with n estimators = 1 and vary the max depth of the tree. Report the mean cross-validated accuracy and accuracy on the test set for the best parameters. Plot the decision boundaries for each dataset.
    n_estimator_pool = [1,100]
    parameters = {'max_depth': [1, 5, 10, 20, 50, 75, 100]}
    for n_estimator in n_estimator_pool:
        print("n_estimator = ", n_estimator)
        rf = RandomForestClassifier(n_estimators = n_estimator)  
        clf = GridSearchCV(rf, parameters)
        clf.fit(X_train, y_train)
        print("Mean Cross Validated Accuracy for the best parameters:",clf.best_score_)
    
    # Accuracy on the test set for the best parameters
        test_score = clf.score(X_test, y_test)
        print(f"Dataset {idx}: {clf.best_params_}")
        print("Accuracy on Test Set for the best parameters:", test_score)
        
    #TODO plot decision boundary
        plt.figure()
        plotting.plot_decision_boundary(X_train, clf)
        plotting.plot_dataset(X_train, X_test, y_train, y_test)
        plt.title(f"Dataset {idx} with n_estimator = {n_estimator}")
        plt.savefig(f'../plots/task3_1_decision_boundary_dataset_{idx}_n_estimator_{n_estimator}.jpg', dpi=120)
        plt.show()
    # plt.close()
    
    