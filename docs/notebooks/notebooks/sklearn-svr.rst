
sklearn: SVM regression
=======================

In this example we will show how to use Optunity to tune hyperparameters
for support vector regression, more specifically:

-  measure empirical improvements through nested cross-validation

-  optimizing hyperparameters for a given family of kernel functions

-  determining the optimal model without choosing the kernel in advance

.. code:: python

    import math
    import itertools
    import optunity
    import optunity.metrics
    import sklearn.svm
We start by creating the data set. We use sklearn's diabetes data.

.. code:: python

    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    n = diabetes.data.shape[0]
    
    data = diabetes.data
    targets = diabetes.target
Nested cross-validation
-----------------------

Nested cross-validation is used to estimate generalization performance
of a full learning pipeline, which includes optimizing hyperparameters.
We will use three folds in the outer loop.

When using default hyperparameters, there is no need for an inner
cross-validation procedure. However, if we want to include tuning in the
learning pipeline, the inner loop is used to determine generalization
performance with optimized hyperparameters.

We start by measuring generalization performance with default
hyperparameters.

.. code:: python

    # we explicitly generate the outer_cv decorator so we can use it twice
    outer_cv = optunity.cross_validated(x=data, y=targets, num_folds=3)
    
    def compute_mse_standard(x_train, y_train, x_test, y_test):
        """Computes MSE of an SVR with RBF kernel and default hyperparameters.
        """
        model = sklearn.svm.SVR().fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)
    
    # wrap with outer cross-validation
    compute_mse_standard = outer_cv(compute_mse_standard)
``compute_mse_standard()`` returns a three-fold cross-validation
estimate of MSE for an SVR with default hyperparameters.

.. code:: python

    compute_mse_standard()



.. parsed-literal::

    6190.481497665955



We will create a function that returns MSE based on optimized
hyperparameters, where we choose a polynomial kernel in advance.

.. code:: python

    def compute_mse_poly_tuned(x_train, y_train, x_test, y_test):
        """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""
    
        # define objective function for tuning
        @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
        def tune_cv(x_train, y_train, x_test, y_test, C, degree, coef0):
            model = sklearn.svm.SVR(C=C, degree=degree, coef0=coef0, kernel='poly').fit(x_train, y_train)
            predictions = model.predict(x_test)
            return optunity.metrics.mse(y_test, predictions)
    
        # optimize parameters
        optimal_pars, _, _ = optunity.minimize(tune_cv, 150, C=[1000, 20000], degree=[2, 5], coef0=[0, 1])
        print("optimal hyperparameters: " + str(optimal_pars))
    
        tuned_model = sklearn.svm.SVR(kernel='poly', **optimal_pars).fit(x_train, y_train)
        predictions = tuned_model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)
    
    # wrap with outer cross-validation
    compute_mse_poly_tuned = outer_cv(compute_mse_poly_tuned)
``compute_mse_poly_tuned()`` returns a three-fold cross-validation
estimate of MSE for an SVR with RBF kernel with tuned hyperparameters
:math:`1000 < C < 20000`, :math:`2 < degree < 5` and
:math:`0 < coef0 < 1` with a budget of 150 function evaluations. Each
tuple of hyperparameters is evaluated using twice-iterated 5-fold
cross-validation.

.. code:: python

    compute_mse_poly_tuned()

.. parsed-literal::

    optimal hyperparameters: {'C': 12078.673881034498, 'coef0': 0.5011052085197018, 'degree': 4.60890281463418}
    optimal hyperparameters: {'C': 14391.165364583334, 'coef0': 0.17313151041666666, 'degree': 2.35826171875}
    optimal hyperparameters: {'C': 11713.456382191061, 'coef0': 0.49836486667796476, 'degree': 4.616077904035152}




.. parsed-literal::

    3047.035965991627



The polynomial kernel yields pretty good results when optimized, but
maybe we can do even better with an RBF kernel.

.. code:: python

    def compute_mse_rbf_tuned(x_train, y_train, x_test, y_test):
        """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""
    
        # define objective function for tuning
        @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
        def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
            model = sklearn.svm.SVR(C=C, gamma=gamma).fit(x_train, y_train)
            predictions = model.predict(x_test)
            return optunity.metrics.mse(y_test, predictions)
    
        # optimize parameters
        optimal_pars, _, _ = optunity.minimize(tune_cv, 150, C=[1, 100], gamma=[0, 50])
        print("optimal hyperparameters: " + str(optimal_pars))
    
        tuned_model = sklearn.svm.SVR(**optimal_pars).fit(x_train, y_train)
        predictions = tuned_model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)
    
    # wrap with outer cross-validation
    compute_mse_rbf_tuned = outer_cv(compute_mse_rbf_tuned)
``compute_mse_rbf_tuned()`` returns a three-fold cross-validation
estimate of MSE for an SVR with RBF kernel with tuned hyperparameters
:math:`1 < C < 100` and :math:`0 < \gamma < 5` with a budget of 150
function evaluations. Each tuple of hyperparameters is evaluated using
twice-iterated 5-fold cross-validation.

.. code:: python

    compute_mse_rbf_tuned()

.. parsed-literal::

    optimal hyperparameters: {'C': 21.654003906250026, 'gamma': 16.536188056152554}
    optimal hyperparameters: {'C': 80.89867187499999, 'gamma': 3.2346692538501784}
    optimal hyperparameters: {'C': 19.35431640625002, 'gamma': 22.083848774716085}




.. parsed-literal::

    2990.8572696483493



Woop! Seems like an RBF kernel is a good choice. An optimized RBF kernel
leads to a 50% reduction in MSE compared to the default configuration.

Determining the kernel family during tuning
-------------------------------------------

In the previous part we've seen that the choice of kernel and its
parameters significantly impact performance. However, testing every
kernel family separately is cumbersome. It's better to let Optunity do
the work for us.

Optunity can optimize conditional search spaces, here the kernel family
and depending on which family the hyperparameterization (:math:`\gamma`,
degree, coef0, ...). We start by defining the search space (we will try
the linear, polynomial and RBF kernel).

.. code:: python

    space = {'kernel': {'linear': {'C': [0, 100]},
                        'rbf': {'gamma': [0, 50], 'C': [1, 100]},
                        'poly': {'degree': [2, 5], 'C': [1000, 20000], 'coef0': [0, 1]}
                        }
             }
Now we do nested cross-validation again.

.. code:: python

    def compute_mse_all_tuned(x_train, y_train, x_test, y_test):
        """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""
    
        # define objective function for tuning
        @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
        def tune_cv(x_train, y_train, x_test, y_test, kernel, C, gamma, degree, coef0):
            if kernel == 'linear':
                model = sklearn.svm.SVR(kernel=kernel, C=C)
            elif kernel == 'poly':
                model = sklearn.svm.SVR(kernel=kernel, C=C, degree=degree, coef0=coef0)
            elif kernel == 'rbf':
                model = sklearn.svm.SVR(kernel=kernel, C=C, gamma=gamma)
            else: 
                raise ArgumentError("Unknown kernel function: %s" % kernel)
            model.fit(x_train, y_train)
    
            predictions = model.predict(x_test)
            return optunity.metrics.mse(y_test, predictions)
    
        # optimize parameters
        optimal_pars, _, _ = optunity.minimize_structured(tune_cv, num_evals=150, search_space=space)
        
        # remove hyperparameters with None value from optimal pars
        for k, v in optimal_pars.items():
            if v is None: del optimal_pars[k]
        print("optimal hyperparameters: " + str(optimal_pars))
        
        tuned_model = sklearn.svm.SVR(**optimal_pars).fit(x_train, y_train)
        predictions = tuned_model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)
    
    # wrap with outer cross-validation
    compute_mse_all_tuned = outer_cv(compute_mse_all_tuned)
And now the kernel family will be optimized along with its
hyperparameterization.

.. code:: python

    compute_mse_all_tuned()

.. parsed-literal::

    optimal hyperparameters: {'kernel': 'rbf', 'C': 33.70116043112164, 'gamma': 16.32317353448437}
    optimal hyperparameters: {'kernel': 'rbf', 'C': 58.11404170763237, 'gamma': 26.45349823062099}
    optimal hyperparameters: {'kernel': 'poly', 'C': 14964.421875843143, 'coef0': 0.5127175861493205, 'degree': 4.045210787998622}




.. parsed-literal::

    3107.625560844859



It looks like the RBF and polynomial kernel are competitive for this
problem.
