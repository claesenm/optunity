# Example of tuning an SVR model in scikit-learn with Optunity
# This example requires sklearn
import math
import itertools
import optunity
import optunity.metrics
import sklearn.svm
import matplotlib.pylab as plt
import time

# CREATE THE TRAINING SET
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
n = diabetes.data.shape[0]

data = diabetes.data
targets = diabetes.target

# we will use nested 3-fold cross-validation
# in the outer cross-validation pmseedure
# we make the decorator explicitly so we can reuse the same folds
# in both tuned and untuned approaches
outer_cv = optunity.cross_validated(x=data, y=targets, num_folds=3)

# compute area under mse curve of default parameters
def compute_mse_standard(x_train, y_train, x_test, y_test):
    model = sklearn.svm.SVR().fit(x_train, y_train)
    predictions = model.predict(x_test)
    return optunity.metrics.mse(y_test, predictions)

# decorate with cross-validation
compute_mse_standard = outer_cv(compute_mse_standard)
mse_standard = compute_mse_standard()
print('Nested cv mean squared error of non-tuned model: ' + str(mse_standard))

# compute area under mse curve with tuned parameters
# we use 2x5 fold cross-validation while tuning
def compute_mse_tuned(x_train, y_train, x_test, y_test):

    # define objective function
    @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
    def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
        model = sklearn.svm.SVR(C=C, gamma=gamma).fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)

    # optimize parameters
    optimal_pars, _, _ = optunity.minimize(tune_cv, 200, C=[0, 10], gamma=[0, 10], pmap=optunity.pmap)
    # if you are running this in IPython, optunity.pmap will not work
    # more info at: https://github.com/claesenm/optunity/issues/8
    # comment out the above line and replace by the one below:
    # optimal_pars, _, _ = optunity.minimize(inner_cv, 150, C=[0, 10], gamma=[0, 0.1])

    tuned_model = sklearn.svm.SVR(**optimal_pars).fit(x_train, y_train)
    predictions = tuned_model.predict(x_test)
    return optunity.metrics.mse(y_test, predictions)

# decorate with cross-validation
compute_mse_tuned = outer_cv(compute_mse_tuned)

t = time.time()
mse_tuned = compute_mse_tuned()
diff = time.time() - t
print('Nested cv mean squared error of tuned model: ' + str(mse_tuned))
print('Tuning time (approx): ' + str(diff/3) + ' seconds') # we tuned 3 times


# generate folds, so we know the indices of test instances at any point
folds = optunity.generate_folds(data.shape[0], num_folds=3)

# create another cross-validation decorator
# we will compare nested cross-validation results for both tuned and untuned models
# to do this, we will perform  nested cross-validation but aggregate results using the identity function
# this will yield the predictions
outer_cv = optunity.cross_validated(x=data, y=targets, num_folds=3, folds=[folds],
                                    aggregator=optunity.cross_validation.identity)

def svr_untuned_predictions(x_train, y_train, x_test, y_test):
    model = sklearn.svm.SVR().fit(x_train, y_train)
    return model.predict(x_test).tolist()


def svr_tuned_predictions(x_train, y_train, x_test, y_test):
    @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
    def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
        model = sklearn.svm.SVR(C=C, gamma=gamma).fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)

    optimal_pars, _, _ = optunity.minimize(tune_cv, 200, C=[0, 20],
                                           gamma=[0, 10], pmap=optunity.pmap)
    tuned_model = sklearn.svm.SVR(**optimal_pars).fit(x_train, y_train)
    return tuned_model.predict(x_test).tolist()

svr_untuned_predictions = outer_cv(svr_untuned_predictions)
svr_tuned_predictions = outer_cv(svr_tuned_predictions)

untuned_preds = svr_untuned_predictions()
tuned_preds = svr_tuned_predictions()

true_targets = [targets[i] for i in itertools.chain(*folds)]
untuned = list(itertools.chain(*untuned_preds))
tuned = list(itertools.chain(*tuned_preds))

#for y, u, t in zip(true_targets, untuned, tuned):
#    print(str(y) + ' :: ' + str(u) + ' :: ' + str(t))

print('plotting results')

plt.plot(range(len(true_targets)), sorted(map(lambda x, y: math.fabs(x-y), tuned, true_targets)), 'b')
plt.plot(range(len(true_targets)), sorted(map(lambda x, y: math.fabs(x-y), untuned, true_targets)), 'r')
plt.xlabel('k largest error')
plt.ylabel('absolute error')
plt.legend(['tuned model', 'default hyperparameters'])
plt.show()
