from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


def calculate_mse(targets, predictions):
    """
    :param targets:
    :param predictions: Predictions obtained by using the model
    :return:
    """
    mse = mean_squared_error(y_pred=predictions, y_true=targets)
    return mse


def solve_regression_task(features, targets):
    """
    :param features:
    :param targets:
    :return: 
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    n_hidden_neurons_list = [40, 48, 56] # TODO (try at least 3 different numbers of neurons)

    # Using GridSearch
    parameters = {'alpha': [0.002, 0.003, 0.004], 'hidden_layer_sizes': n_hidden_neurons_list,
                  'learning_rate_init':[0.0003, 0.0005, 0.0008], \
                  'activation': ['relu', 'logistic'], 'solver':['adam', 'lbfgs', 'sgd']}
    nn = MLPRegressor(random_state=1, max_iter=3000, activation='relu', learning_rate='constant')
    clf = GridSearchCV(estimator=nn, param_grid=parameters,n_jobs=-1)
    clf.fit(X_train, y_train)

    print(f'best score: {clf.best_score_}')
    print(f'grid search selected params: {clf.best_params_}')
    
    # Calculate predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    print(f'Train MSE: {calculate_mse(y_train, y_pred_train):.4f}. Test MSE: {calculate_mse(y_test, y_pred_test):.4f}')

    
