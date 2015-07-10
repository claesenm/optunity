
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

    6100.237626940242



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

    optimal hyperparameters: {'C': 19868.74221087047, 'coef0': 0.45679573424872666, 'degree': 4.879991987802887}
    optimal hyperparameters: {'C': 14136.347474716942, 'coef0': 0.13972510935921983, 'degree': 2.7560546874999994}
    optimal hyperparameters: {'C': 13402.57161458333, 'coef0': 0.4447460937500003, 'degree': 3.312242362741492}




.. parsed-literal::

    3095.2745346802844



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

    optimal hyperparameters: {'C': 19.08662689915651, 'gamma': 17.702242473189145}
    optimal hyperparameters: {'C': 67.48642799529746, 'gamma': 6.483072916666674}
    optimal hyperparameters: {'C': 39.601226927358795, 'gamma': 10.795909203898383}




.. parsed-literal::

    3060.918626604345



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

::


    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)

    <ipython-input-15-d88731cf8f17> in <module>()
    ----> 1 compute_mse_all_tuned()
    

    /data/svn/claesenm/python/optunity/optunity/cross_validation.pyc in __call__(self, *args, **kwargs)
        401                     kwargs['y_train'] = select(self.y, rows_train)
        402                     kwargs['y_test'] = select(self.y, rows_test)
    --> 403                 scores.append(self.f(**kwargs))
        404         return self.reduce(scores)
        405 


    <ipython-input-14-1ef61cf64223> in compute_mse_all_tuned(x_train, y_train, x_test, y_test)
         18 
         19     # optimize parameters
    ---> 20     optimal_pars, _, _ = optunity.minimize_structured(tune_cv, num_evals=150, search_space=space)
         21     print("optimal hyperparameters: " + str(optimal_pars))
         22 


    /data/svn/claesenm/python/optunity/optunity/api.pyc in minimize_structured(f, search_space, num_evals, pmap)
        398     solver = make_solver(**suggestion)
        399     solution, details = optimize(solver, f, maximize=False, max_evals=num_evals,
    --> 400                                  pmap=pmap, decoder=tree.decode)
        401     return solution, details, suggestion
        402 


    /data/svn/claesenm/python/optunity/optunity/api.pyc in optimize(solver, func, maximize, max_evals, pmap, decoder)
        243     time = timeit.default_timer()
        244     try:
    --> 245         solution, report = solver.optimize(f, maximize, pmap=pmap)
        246     except fun.MaximumEvaluationsException:
        247         # early stopping because maximum number of evaluations is reached


    /data/svn/claesenm/python/optunity/optunity/solvers/ParticleSwarm.pyc in optimize(self, f, maximize, pmap)
        268 
        269         for g in range(self.num_generations):
    --> 270             fitnesses = pmap(evaluate, list(map(self.particle2dict, pop)))
        271             for part, fitness in zip(pop, fitnesses):
        272                 part.fitness = fit * util.score(fitness)


    /data/svn/claesenm/python/optunity/optunity/solvers/ParticleSwarm.pyc in evaluate(d)
        257         @functools.wraps(f)
        258         def evaluate(d):
    --> 259             return f(**d)
        260 
        261         if maximize:


    /data/svn/claesenm/python/optunity/optunity/functions.pyc in wrapped_f(*args, **kwargs)
        341             else:
        342                 wrapped_f.num_evals += 1
    --> 343                 return f(*args, **kwargs)
        344         wrapped_f.num_evals = 0
        345         return wrapped_f


    /data/svn/claesenm/python/optunity/optunity/constraints.pyc in wrapped_f(*args, **kwargs)
        148         def wrapped_f(*args, **kwargs):
        149             try:
    --> 150                 return f(*args, **kwargs)
        151             except ConstraintViolation:
        152                 return default


    /data/svn/claesenm/python/optunity/optunity/constraints.pyc in wrapped_f(*args, **kwargs)
        126             if violations:
        127                 raise ConstraintViolation(violations, *args, **kwargs)
    --> 128             return f(*args, **kwargs)
        129         wrapped_f.constraints = constraints
        130         return wrapped_f


    /data/svn/claesenm/python/optunity/optunity/constraints.pyc in func(*args, **kwargs)
        263         @functools.wraps(f)
        264         def func(*args, **kwargs):
    --> 265             return f(*args, **kwargs)
        266     return func
        267 


    /data/svn/claesenm/python/optunity/optunity/search_spaces.pyc in wrapped(**kwargs)
        223         def wrapped(**kwargs):
        224             decoded = self.decode(kwargs)
    --> 225             return f(**decoded)
        226         return wrapped
        227 


    /data/svn/claesenm/python/optunity/optunity/functions.pyc in wrapped_f(*args, **kwargs)
        286         value = wrapped_f.call_log.get(*args, **kwargs)
        287         if value is None:
    --> 288             value = f(*args, **kwargs)
        289             wrapped_f.call_log.insert(value, *args, **kwargs)
        290         return value


    /data/svn/claesenm/python/optunity/optunity/cross_validation.pyc in __call__(self, *args, **kwargs)
        401                     kwargs['y_train'] = select(self.y, rows_train)
        402                     kwargs['y_test'] = select(self.y, rows_test)
    --> 403                 scores.append(self.f(**kwargs))
        404         return self.reduce(scores)
        405 


    <ipython-input-14-1ef61cf64223> in tune_cv(x_train, y_train, x_test, y_test, kernel, C, gamma, degree, coef0)
         14             raise ArgumentError("Unknown kernel function: %s" % kernel)
         15 
    ---> 16         predictions = model.predict(x_test)
         17         return optunity.metrics.mse(y_test, predictions)
         18 


    /usr/lib/python2.7/dist-packages/sklearn/svm/base.pyc in predict(self, X)
        280         y_pred : array, shape (n_samples,)
        281         """
    --> 282         X = self._validate_for_predict(X)
        283         predict = self._sparse_predict if self._sparse else self._dense_predict
        284         return predict(X)


    /usr/lib/python2.7/dist-packages/sklearn/svm/base.pyc in _validate_for_predict(self, X)
        383     def _validate_for_predict(self, X):
        384         X = atleast2d_or_csr(X, dtype=np.float64, order="C")
    --> 385         if self._sparse and not sp.isspmatrix(X):
        386             X = sp.csr_matrix(X)
        387         if self._sparse:


    AttributeError: 'SVR' object has no attribute '_sparse'

