
Basic: nested cross-validation
==============================

In this notebook we will briefly illustrate how to use Optunity for
nested cross-validation.

Nested cross-validation is used to reliably estimate generalization
performance of a learning pipeline (which may involve preprocessing,
tuning, model selection, ...). Before starting this tutorial, we
recommend making sure you are reliable with basic cross-validation in
Optunity.

We will use a scikit-learn SVM to illustrate the key concepts on the
MNIST data set.

.. code:: python

    import optunity
    import optunity.cross_validation
    import optunity.metrics
    import numpy as np
    import sklearn.svm
We load the digits data set and will construct models to distinguish
digits 6 from and 8.

.. code:: python

    from sklearn.datasets import load_digits
    digits = load_digits()
    n = digits.data.shape[0]
    
    positive_digit = 6
    negative_digit = 8
    
    positive_idx = [i for i in range(n) if digits.target[i] == positive_digit]
    negative_idx = [i for i in range(n) if digits.target[i] == negative_digit]
    
    # add some noise to the data to make it a little challenging
    original_data = digits.data[positive_idx + negative_idx, ...]
    data = original_data + 5 * np.random.randn(original_data.shape[0], original_data.shape[1])
    labels = [True] * len(positive_idx) + [False] * len(negative_idx)
The basic nested cross-validation scheme involves two cross-validation
routines:

-  outer cross-validation: to estimate the generalization performance of
   the learning pipeline. We will use 5folds.

-  inner cross-validation: to use while optimizing hyperparameters. We
   will use twice iterated 10-fold cross-validation.

Here, we have to take into account that we need to stratify the data
based on the label, to ensure we don't run into situations where only
one label is available in the train or testing splits. To do this, we
use the ``strata_by_labels`` utility function.

We will use an SVM with RBF kernel and optimize gamma on an exponential
grid :math:`10^-5 < \gamma < 10^1` and :math:`0< C < 10` on a linear
grid.

.. code:: python

    # outer cross-validation to estimate performance of whole pipeline
    @optunity.cross_validated(x=data, y=labels, num_folds=5,
                              strata=optunity.cross_validation.strata_by_labels(labels))
    def nested_cv(x_train, y_train, x_test, y_test):
    
        # inner cross-validation to estimate performance of a set of hyperparameters
        @optunity.cross_validated(x=x_train, y=y_train, num_folds=10, num_iter=2,
                                  strata=optunity.cross_validation.strata_by_labels(y_train))
        def inner_cv(x_train, y_train, x_test, y_test, C, logGamma):
            # note that the x_train, ... variables in this function are not the same
            # as within nested_cv!
            model = sklearn.svm.SVC(C=C, gamma=10 ** logGamma).fit(x_train, y_train)
            predictions = model.decision_function(x_test)
            return optunity.metrics.roc_auc(y_test, predictions)
    
        hpars, info, _ = optunity.maximize(inner_cv, num_evals=100, 
                                        C=[0, 10], logGamma=[-5, 1])
        print('')
        print('Hyperparameters: ' + str(hpars))
        print('Cross-validated AUROC after tuning: %1.3f' % info.optimum)
        model = sklearn.svm.SVC(C=hpars['C'], gamma=10 ** hpars['logGamma']).fit(x_train, y_train)
        predictions = model.decision_function(x_test)
        return optunity.metrics.roc_auc(y_test, predictions)
    
    auc = nested_cv()
    print('')
    print('Nested AUROC: %1.3f' % auc)

.. parsed-literal::

    
    Hyperparameters: {'logGamma': -3.8679410473451057, 'C': 0.6162109374999996}
    Cross-validated AUROC after tuning: 1.000
    
    Hyperparameters: {'logGamma': -4.535231399331072, 'C': 0.4839113474508706}
    Cross-validated AUROC after tuning: 0.999
    
    Hyperparameters: {'logGamma': -4.0821875, 'C': 1.5395986549905802}
    Cross-validated AUROC after tuning: 1.000
    
    Hyperparameters: {'logGamma': -3.078125, 'C': 6.015625}
    Cross-validated AUROC after tuning: 1.000
    
    Hyperparameters: {'logGamma': -4.630859375, 'C': 3.173828125}
    Cross-validated AUROC after tuning: 1.000
    
    Nested AUROC: 0.999


If you want to explicitly retain statistics from the inner
cross-validation procedure, such as the ones we printed below, we can do
so by returning tuples in the outer cross-validation and using the
``identity`` aggregator.

.. code:: python

    # outer cross-validation to estimate performance of whole pipeline
    @optunity.cross_validated(x=data, y=labels, num_folds=5,
                              strata=optunity.cross_validation.strata_by_labels(labels),
                              aggregator=optunity.cross_validation.identity)
    def nested_cv(x_train, y_train, x_test, y_test):
    
        # inner cross-validation to estimate performance of a set of hyperparameters
        @optunity.cross_validated(x=x_train, y=y_train, num_folds=10, num_iter=2,
                                  strata=optunity.cross_validation.strata_by_labels(y_train))
        def inner_cv(x_train, y_train, x_test, y_test, C, logGamma):
            # note that the x_train, ... variables in this function are not the same
            # as within nested_cv!
            model = sklearn.svm.SVC(C=C, gamma=10 ** logGamma).fit(x_train, y_train)
            predictions = model.decision_function(x_test)
            return optunity.metrics.roc_auc(y_test, predictions)
    
        hpars, info, _ = optunity.maximize(inner_cv, num_evals=100, 
                                        C=[0, 10], logGamma=[-5, 1])
        model = sklearn.svm.SVC(C=hpars['C'], gamma=10 ** hpars['logGamma']).fit(x_train, y_train)
        predictions = model.decision_function(x_test)
        
        # return AUROC, optimized hyperparameters and AUROC during hyperparameter search
        return optunity.metrics.roc_auc(y_test, predictions), hpars, info.optimum
    
    nested_cv_result = nested_cv()
We can then process the results like this:

.. code:: python

    aucs, hpars, optima = zip(*nested_cv_result)
    
    print("AUCs: " + str(aucs))
    print('')
    print("hpars: " + "\n".join(map(str, hpars)))
    print('')
    print("optima: " + str(optima))
    
    mean_auc = sum(aucs) / len(aucs)
    print('')
    print("Mean AUC %1.3f" % mean_auc)

.. parsed-literal::

    AUCs: (0.9992063492063492, 1.0, 1.0, 0.9976190476190476, 0.9984126984126984)
    
    hpars: {'logGamma': -3.5753515625, 'C': 3.9048828125000004}
    {'logGamma': -2.6765234375, 'C': 6.9193359375000005}
    {'logGamma': -3.0538671875, 'C': 2.2935546875}
    {'logGamma': -3.593515625, 'C': 4.4136718749999995}
    {'logGamma': -3.337747403818736, 'C': 4.367953383541078}
    
    optima: (0.9995032051282051, 0.9985177917320774, 0.9994871794871795, 0.9995238095238095, 0.9995032051282051)
    
    Mean AUC 0.999

