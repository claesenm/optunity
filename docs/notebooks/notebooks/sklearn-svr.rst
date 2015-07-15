
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
    import matplotlib.pylab as plt
    import time
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

    6056.311654865084



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

    optimal hyperparameters: {'C': 8103.600260416671, 'coef0': 0.5978059895833331, 'degree': 4.797399399165802}
    optimal hyperparameters: {'C': 19755.083174237032, 'coef0': 0.4450483090495349, 'degree': 4.6829006570233025}
    optimal hyperparameters: {'C': 8680.403645833334, 'coef0': 0.47259114583333317, 'degree': 3.0486328125000006}




.. parsed-literal::

    3122.5387612208156



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

    optimal hyperparameters: {'C': 81.98731244487078, 'gamma': 4.895475395394894}
    optimal hyperparameters: {'C': 52.067311507783465, 'gamma': 6.5244315086089815}
    optimal hyperparameters: {'C': 26.993648162210402, 'gamma': 25.997204804202134}




.. parsed-literal::

    2982.6835697931674



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

    optimal hyperparameters: {'kernel': 'rbf', 'C': 36.396896096116805, 'gamma': 10.015489219932745}
    optimal hyperparameters: {'kernel': 'rbf', 'C': 25.6858037455125, 'gamma': 17.48771066406458}
    optimal hyperparameters: {'kernel': 'rbf', 'C': 93.31245007796964, 'gamma': 7.082799132257288}




.. parsed-literal::

    3001.3258583571483



Looks like an RBF kernel was indeed the best choice!
