===================
Cross-validation
===================

Optunity offers a simple interface to k-fold cross-validation_. This is a statistical approach to measure a model's generalization performance. 
In the context of hyperparameter search, cross-validation is used to estimate the performance of a hyperparameter tuple. The cross-validation routines we offer
are optional and can be replaced by comparable routines from other packages or some other method to estimate generalization performance.

.. _cross-validation: http://en.wikipedia.org/wiki/Cross-validation_(statistics)

The fold generation procedure in Optunity allows for iterated cross-validation and is aware of both strata (data instances that must be spread across folds) 
and clusters (sets of instances that must assigned to a single fold). Please refer to func:`optunity.cross_validated` for implementation and API details.

We will build examples step by step. The basic setup is a ``train`` and ``predict``
function along with some ``data`` to construct folds over::

    from __future__ import print_function
    import optunity as opt

    def train(x, y, filler=''):
        print(filler + 'Training data:')
        for instance, label in zip(x, y):
            print(filler + str(instance) + ' ' + str(label))

    def predict(x, filler=''):
        print(filler + 'Testing data:')
        for instance in x:
            print(filler + str(instance))

    data = list(range(9))
    labels = [0] * 9

The recommended way to perform cross-validation is using the :func:`optunity.cross_validation.cross_validated` function decorator. To use it, you must specify 
an objective function. This function should contain the logic that is placed in the inner loop in cross-validation (e.g. train a model, predict test set, compute score), 
with the following signature: ``f(x_train, y_train, x_test, y_test, hyperpar_1, hyperpar_2, ...)`` (argument names are important):

-   `x_train`: training data
-   `y_train`: training labels (optional)
-   `x_test`: test data
-   `y_test`: test labels (optional)
-   `hyperparameter names`: the hyperparameters that must be optimized

The `cross_validated` decorator takes care of generating folds, partitioning the data, iterating over folds and aggregating partial results. After decoration,
the arguments `x_train`, `y_train`, `x_test` and `y_test` will be bound (e.g. the decorated function does not take these as arguments). The decorated function will
have hyperparameters as (keyword) arguments and returns a cross-validation result.

A simple code example::

    .. highlight:: python

    @opt.cross_validated(x=data, y=labels, num_folds=3)
    def cved(x_train, y_train, x_test, y_test):
        train(x_train, y_train)
        predict(x_test)
        return 0.0

    cved()

If you want to compare different aspects of the learning approach (learning algorithms, score function, ...), 
it is a good idea to use the same cross-validation folds. This is very easy in Optunity by using the `cross_validated` decorator without syntactic sugar. 
Lets say we want to compare an SVM with RBF kernel and polynomial kernel with the same cross-validation configuration:: 

    .. highlight:: python

    import sklearn.svm as svm

    def svm rbf(x_train, y_train, x_test, y_test, C, gamma):
        model = svm.SVC(kernel='rbf', C=C, gamma=gamma).fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return opt.score_functions.accuracy(y_test, y_pred)

    def svm poly(x_train, y_train, x_test, y_test, C, d):
        model = svm.SVC(kernel=’poly’, C=C, degree=d).fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return opt.score_functions.accuracy(y_test, y_pred)

    cv_decorator = opt.cross_validated(x=data, y=labels, num_folds=3)

    svm_rbf_cv = cv_decorator(svm_rbf)
    svm_poly_cv = cv_decorator(svm_poly)

In this example, the function `svm_rbf_cv` takes keyword arguments `C` and `gamma` while `svm_poly_cv` takes `C` and `d`. Both perform cross-validation
on the same data, using the same folds.


Nested cross-validation
--------------------------

Nested cross-validation is a commonly used approach to estimate the generalization 
performance of a modeling process which includes model selection internally. 
A good summary is provided here_.

.. _here: http://stats.stackexchange.com/a/65156/25433

Nested cv consists of two cross-validation procedures wrapped around eachother. The inner cv is
used for model selection, the outer cv estimates generalization performance.



This can be done in a straightforward manner using Optunity::

    @opt.cross_validated(x=data, y=labels, num_folds=3)
    def nested_cv(x_train, y_train, x_test, y_test):

        @opt.cross_validated(x=x_train, y=y_train, num_folds=3)
        def inner_cv(x_train, y_train, x_test, y_test):
            train(x_train, y_train, '...')
            predict(x_test, '...')
            return 0.0

        inner_cv()
        predict(x_test)
        return 0.0

    nested_cv()

The inner :func:`optunity.cross_validated` decorator has access to
the train and test folds generated by the outer procedure (``x_train`` and ``x_test``).
For notational simplicity we assume a problem without labels here.

.. note::
    The inner folds are regenerated in every iteration (since we are redefining ``inner_cv`` each time). 
    The inner folds will therefore be different each time. The outer folds remain static, unless ``regenerate_folds=True`` is passed.

Below we illustrate a more complete example of nested cv, which includes hyperparameter
optimization with :func:`optunity.maximize`. Assume we have access to the following functions
``svm=svm_train(x, y, c, g)`` and ``predictions=svm_predict(svm, x)``. Where ``c`` and ``g``
are hyperparameters to be optimized for accuracy::

    @opt.cross_validated(x=data, y=labels, num_folds=3)
    def nested_cv(x_train, y_train, x_test, y_test):

        @opt.cross_validated(x=x_train, y=y_train, num_folds=3)
        def inner_cv(x_train, y_train, x_test, y_test, c, g):
            svm = svm_train(x_train, y_train, c, g)
            predictions = svm_predict(svm, x_test)
            return opt.score_functions.accuracy(y_test, predictions)

        optimal_parameters, _, _ = opt.maximize(inner_cv, num_evals=100, c=[0, 10], g=[0, 10])
        optimal_svm = svm_train(x_train, y_train, **optimal_parameters)
        predictions = svm_predict(optimal_svm, x_test)
        return opt.score_functions.accuracy(y_test, predictions)

    overall_accuracy = nested_cv()

.. note::
    You are free to use different score and aggregation functions in the inner and outer cv.
