import optunity
import optunity.score_functions
import sklearn.svm
import numpy as np

npos = 200
nneg = 200
d = 1
scale = 10

# train data
train_data = np.vstack([np.random.randn(npos, 2) + np.array([[d, 0.0]] * npos),
                        np.random.randn(nneg, 2) - np.array([[d, 0.0]] * nneg)])
train_labels = [1] * npos + [-1] * nneg

# test data
test_data = np.vstack([np.random.randn(npos, 2) + np.array([[d, 0.0]] * npos),
                       np.random.randn(nneg, 2) - np.array([[d, 0.0]] * nneg)])
test_labels = [1] * npos + [-1] * nneg

train_data = scale * train_data
test_data = scale * test_data

# SVM with default parameter settings
default_model = sklearn.svm.SVC().fit(train_data, train_labels)
default_predictions = default_model.predict(test_data)
default_accuracy = optunity.score_functions.accuracy(test_labels, default_predictions)
print('test accuracy with default hyperparameters: ' + str(default_accuracy))

@optunity.cross_validated(x=train_data, y=train_labels, num_folds=5, num_iter=2)
def svm_acc(x_train, y_train, x_test, y_test, C, gamma):
    model = sklearn.svm.SVC(C=C, gamma=gamma).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return optunity.score_functions.accuracy(y_test, y_pred)

def grid_search(gammas, Cs):
    best_acc = 0.0
    best_gamma = None
    best_C = None
    for C in Cs:
        for gamma in gammas:
            acc = svm_acc(C=C, gamma=gamma)
            if acc > best_acc:
                best_acc = acc
                best_gamma = gamma
                best_C = C
    return best_acc, best_gamma, best_C

grid_acc, grid_gamma, grid_c = grid_search([0.001, 0.01, 0.1, 1, 10],
                                           [0.001, 0.01, 0.1, 1, 10])

optimal_pars, details, _ = optunity.maximize(svm_acc, num_evals=100, C=[0, 20], gamma=[0, 2])
optimal_model = sklearn.svm.SVC(**optimal_pars).fit(train_data, train_labels)

optimal_predictions = optimal_model.predict(test_data)
optimal_accuracy = optunity.score_functions.accuracy(test_labels, optimal_predictions)
print('test accuracy with optimized hyperparameters: ' + str(optimal_accuracy))
print('optimized hyperparameters: ' + str(optimal_pars))
print('optimization took ' + str(details.stats['time']) + ' seconds')
