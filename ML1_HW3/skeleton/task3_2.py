import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from datasets import get_heart_dataset, get_toy_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
import pickle as pkl
import pandas as pd

if __name__ == '__main__':
  X_train, X_test, y_train, y_test = get_toy_dataset(4)


  #TODO fit a random forest classifier and check how well it performs on the test set after tuning the parameters,
  # report your results
  parameters_rf = {'max_depth': [1, 5, 10, 20, 50, 75, 100, 150, 200]}
  rf = RandomForestClassifier()
  clf_rf = GridSearchCV(rf, parameters_rf)
  clf_rf.fit(X_train, y_train)
  print("Best parameters for Random Forest:",clf_rf.best_params_) 
  rf = RandomForestClassifier(max_depth = clf_rf.best_params_.get("max_depth"))
  rf.fit(X_train,y_train)
  print("Performance of Random Forest Classifier on test set:", rf.score(X_test,y_test))
  
 
  #TODO fit a SVC and find suitable parameters, report your results  
  parameters_svc = {'C': [0.0005, 0.005, 0.1, 0.5, 1.0, 1.5], 'gamma': ["scale", "auto"]}
  svc = SVC()
  clf_svc = GridSearchCV(svc, parameters_svc)
  clf_svc.fit(X_train, y_train)
  print("Best parameters for SVC:",clf_svc.best_params_)
  svc = SVC(C = clf_svc.best_params_.get("C"), gamma = clf_svc.best_params_.get("gamma"))
  svc.fit(X_train,y_train)
  print("Performance of SVC classifier on test set: ", svc.score(X_test,y_test))
  

  # TODO create a bar plot (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html#matplotlib.pyplot.barh)
  # of the `feature_importances_` of the RF classifier.
  feature_names = [f"feature {i}" for i in range(X_train.shape[1])]
  forest_importances = pd.Series(rf.feature_importances_, index=feature_names)
  fig, ax = plt.subplots()
  forest_importances.plot.bar()
  plt.xlabel("Feature Importance")
  fig.tight_layout()
  plt.savefig(f'../plots/task3_2_Feature_Importance.jpg')
    
  # TODO create another RF classifier
  # Use recursive feature elimination (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV)
  # to automatically choose the best number of parameters
  # set `scoring = 'accuracy'` to look for the feature subset with highest accuracy
  # and fit the RFECV to the training data
  rf = RandomForestClassifier()
  clf_rf = GridSearchCV(rf, parameters_rf)
  clf_rf.fit(X_train, y_train) 
  rf = RandomForestClassifier(max_depth = clf_rf.best_params_.get("max_depth"))
  rfecv = RFECV(rf,scoring = "accuracy")
  X_train_new  = rfecv.fit_transform(X_train,y_train)

  # TODO use the RFECV to transform the training and test dataset -- it automatically removes the least important
  # feature columns from the datasets. You don't have to change y_train or y_test
  X_test_new = rfecv.transform(X_test)
  # Fit a SVC classifier on the new dataset. Do you see a difference in performance?
  svc = SVC()
  clf_svc = GridSearchCV(svc, parameters_svc)
  clf_svc.fit(X_train_new, y_train)
  print("Mean Cross Validated Accuracy of best parameters after transformation:",clf_svc.best_score_)
  svc = SVC(C = clf_svc.best_params_.get("C"), gamma = clf_svc.best_params_.get("gamma"))
  svc.fit(X_train_new,y_train)
  print("Accuracy on test set with best parameters after transformation:",svc.score(X_test_new,y_test))

  

  
