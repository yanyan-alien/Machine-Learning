import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import random
import warnings
from array import array
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def reduce_dimension(features, n_components: int):
    """
    :param features: Data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality. Shape: (n_samples, n_components)
    """

    # Create an instance of PCA from sklearn.decomposition (already imported). Set the parameters (some of them are specified in the HW2 sheet).
    pca = PCA(random_state=1, n_components=n_components, whiten=True)
    X_reduced = pca.fit_transform(features) # TODO - Fit the model with features, and apply the transformation on the features.
    explained_var = (pca.explained_variance_ratio_) # Calculate the percentage of variance explained
    
    total_explained_var = 0
    for i in explained_var:
        total_explained_var = total_explained_var + i
        
    print(f'Total Explained variance: {total_explained_var}')
    print(f'Explained variance of each n component: {explained_var}')
    return X_reduced

def train_nn(features, targets):
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons = [2, 10, 100, 200] # TODO create a list
    # TODO create an instance of MLPClassifier from sklearn.neural_network (already imported).
    # Set the parameters (some of them are specified in the HW2 sheet).
    for n_hid in n_hidden_neurons:
        print("----- ", n_hid, "neurons -----")
        clf = MLPClassifier(max_iter=500, solver='adam', random_state=1, hidden_layer_sizes=n_hid).fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        train_acc = clf.score(X_train, y_train)
        loss = clf.loss_ # TODO
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')

def train_nn_with_regularization(features, targets):
    """
    Train MLPClassifier using regularization.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    # Copy your code from train_nn, but experiment now with regularization (alpha, early_stopping).
    n_hidden_neurons = [2, 10, 100, 200]
    for n_hid in n_hidden_neurons:
        print("----- ", n_hid, "neurons -----")
        print('(a) alpha = 0.1')
        clf = MLPClassifier(max_iter=500, solver='adam', random_state=1, hidden_layer_sizes=n_hid, alpha=0.1).fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        train_acc = clf.score(X_train, y_train)
        loss = clf.loss_ # TODO
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')

        print('(b) early_stopping = true')
        clf = MLPClassifier(max_iter=500, solver='adam', random_state=1, hidden_layer_sizes=n_hid, early_stopping=True).fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        train_acc = clf.score(X_train, y_train)
        loss = clf.loss_ # TODO
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')

        print('(c) both -> early_stopping = True, alpha = 0.1')
        clf = MLPClassifier(max_iter=500, solver='adam', random_state=1, hidden_layer_sizes=n_hid, alpha=0.1, early_stopping=True).fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        train_acc = clf.score(X_train, y_train)
        loss = clf.loss_ # TODO
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')

def train_nn_with_different_seeds(features, targets):
    """
    Train MLPClassifier using different seeds.
    Print (mean +/- std) accuracy on the training and test set.
    Print confusion matrix and classification report.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    seeds = [10,20,85,1000,20500]  # TODO create a list of different seeds of your choice
    train_acc_list = []
    test_acc_list = []
    
    for ran_seed in seeds:
    # TODO create an instance of MLPClassifier, check the perfomance for different seeds
        clf = MLPClassifier(max_iter=500, solver='adam', random_state=ran_seed, hidden_layer_sizes=100, alpha=0.1).fit(X_train, y_train)
        print("Seed: ", ran_seed)
        test_acc = clf.score(X_test, y_test)
        train_acc = clf.score(X_train, y_train)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        loss = clf.loss_ # TODO for each seed (for you as a sanity check that the loss stays similar for different seeds, no need to include it in the report)
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')
        
    train_acc_arr = array("f", train_acc_list)
    test_acc_arr = array("f", test_acc_list)
    train_acc_mean = np.mean(train_acc_arr, dtype=np.float64) 
    train_acc_std = np.std(train_acc_arr, dtype=np.float64) 
    test_acc_mean = np.mean(test_acc_arr, dtype=np.float64) 
    test_acc_std = np.std(test_acc_arr, dtype=np.float64)
    print(f'Mean Accuracy on the train set: {train_acc_mean:.4f} +/- {train_acc_std:.4f}')
    print(f'Mean Accuracy on the test set: {test_acc_mean:.4f} +/- {test_acc_std:.4f}')
    
    # TODO: print min and max accuracy as well
    test_acc_min = np.min(test_acc_arr)
    test_acc_max = np.max(test_acc_arr)
    train_acc_min = np.min(train_acc_arr)
    train_acc_max = np.max(train_acc_arr)
    print(f'Min accuracy on the train set: {train_acc_min:.4f} & Max accuracy on the train set: {train_acc_max:.4f}')
    print(f'Min accuracy on the test set: {test_acc_min:.4f} & Max accuracy on the test set: {test_acc_max:.4f}')
    

    # TODO: plot the loss curve
    clf = MLPClassifier(max_iter=500, solver='adam', random_state=85, hidden_layer_sizes=100, alpha=0.1).fit(X_train, y_train)
    iter = clf.n_iter_
    x = []
    for i in range(iter):
        x.append(i) 
    loss_curve = clf.loss_curve_
    plt.plot(x, loss_curve, label='Loss over iterations')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over iteration')
    
    # TODO: Confusion matrix and classification report (for one classifier that performs well)
    print("Predicting on the test set")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred)) 
    print(confusion_matrix(y_test, y_pred, labels=range(10)))


def perform_grid_search(features, targets):

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    parameters = {'activation': ['logistic', 'relu'], 'alpha': [0, 0.1, 1, 10], 'solver': ['adam', 'lbfgs'], 
                      'hidden_layer_sizes': [(100,), (200,)]}

    # nn = # TODO create an instance of MLPClassifier. Do not forget to set parameters as specified in the HW2 sheet.
    # grid_search = # TODO create an instance of GridSearchCV from sklearn.model_selection (already imported) with
    # appropriate params. Set: n_jobs=-1, this is another parameter of GridSearchCV, in order to get faster execution of the code.
    nn = MLPClassifier(max_iter=100, random_state=1, learning_rate_init=0.01)
    clf = GridSearchCV(estimator=nn, param_grid=parameters,n_jobs=-1)
    
    # TODO call fit on the train data
    clf.fit(X_train, y_train)
    # TODO print the best score
    # TODO print the best parameters found by grid_search
    print(clf.best_score_)
    print(clf.best_params_)

    test_acc = clf.score(X_test, y_test)
    train_acc = clf.score(X_train, y_train)
    print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')