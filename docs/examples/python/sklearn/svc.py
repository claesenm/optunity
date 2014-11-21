import optunity
import sklearn.svm

# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=data, y=labels, num_folds=10, num_iter=2)
def svm_auc(x_train, y_train, x_test, y_test, C, gamma):
    model = sklearn.svm.SVC(C=C, gamma=gamma).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return optunity.score_functions.roc_auc(y_test, y_pred)

# perform tuning
optimal_pars, _, _ = optunity.maximize(svm_auc, num_evals=200, C=[0, 10], gamma=[0, 1])

# train model on the full training set with tuned hyperparameters
optimal_model = sklearn.svm.SVC(**optimal_pars).fit(data, labels)
