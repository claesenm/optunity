
sklearn: SVM classification
===========================

In this example we will use Optunity to optimize hyperparameters for a
support vector machine classifier (SVC) in scikit-learn. We will learn a
model to distinguish digits 8 and 9 in the MNIST data set in two
settings

-  tune SVM with RBF kernel
-  tune SVM with RBF, polynomial or linear kernel, that is choose the
   kernel function and its hyperparameters at once

.. code:: python

    import optunity
    import optunity.metrics
    
    # comment this line if you are running the notebook
    import sklearn.svm
    import numpy as np
Create the data set: we use the MNIST data set and will build models to
distinguish digits 8 and 9.

.. code:: python

    from sklearn.datasets import load_digits
    digits = load_digits()
    n = digits.data.shape[0]
    
    positive_digit = 8
    negative_digit = 9
    
    positive_idx = [i for i in range(n) if digits.target[i] == positive_digit]
    negative_idx = [i for i in range(n) if digits.target[i] == negative_digit]
    
    # add some noise to the data to make it a little challenging
    original_data = digits.data[positive_idx + negative_idx, ...]
    data = original_data + 5 * np.random.randn(original_data.shape[0], original_data.shape[1])
    labels = [True] * len(positive_idx) + [False] * len(negative_idx)
First, lets see the performance of an SVC with default hyperparameters.

.. code:: python

    # compute area under ROC curve of default parameters
    @optunity.cross_validated(x=data, y=labels, num_folds=5)
    def svm_default_auroc(x_train, y_train, x_test, y_test):
        model = sklearn.svm.SVC().fit(x_train, y_train)
        decision_values = model.decision_function(x_test)
        auc = optunity.metrics.roc_auc(y_test, decision_values)
        return auc
    
    svm_default_auroc()



.. parsed-literal::

    0.7838539081885857



Tune SVC with RBF kernel 
-------------------------

In order to use Optunity to optimize hyperparameters, we start by
defining the objective function. We will use 5-fold cross-validated area
under the ROC curve. For now, lets restrict ourselves to the RBF kernel
and optimize :math:`C` and :math:`\gamma`.

We start by defining the objective function ``svm_rbf_tuned_auroc()``,
which accepts :math:`C` and :math:`\gamma` as arguments.

.. code:: python

    #we will make the cross-validation decorator once, so we can reuse it later for the other tuning task
    # by reusing the decorator, we get the same folds etc.
    cv_decorator = optunity.cross_validated(x=data, y=labels, num_folds=5)
    
    def svm_rbf_tuned_auroc(x_train, y_train, x_test, y_test, C, gamma):
        model = sklearn.svm.SVC(C=C, gamma=gamma).fit(x_train, y_train)
        decision_values = model.decision_function(x_test)
        auc = optunity.metrics.roc_auc(y_test, decision_values)
        return auc
    
    svm_rbf_tuned_auroc = cv_decorator(svm_rbf_tuned_auroc)
    # this is equivalent to the more common syntax below
    # @optunity.cross_validated(x=data, y=labels, num_folds=5)
    # def svm_rbf_tuned_auroc...
    
    svm_rbf_tuned_auroc(C=1.0, gamma=1.0)



.. parsed-literal::

    0.5



Now we can use Optunity to find the hyperparameters that maximize AUROC.

.. code:: python

    optimal_rbf_pars, info, _ = optunity.maximize(svm_rbf_tuned_auroc, num_evals=150, C=[0, 10], gamma=[0, 0.1])
    # when running this outside of IPython we can parallelize via optunity.pmap
    # optimal_rbf_pars, _, _ = optunity.maximize(svm_rbf_tuned_auroc, 150, C=[0, 10], gamma=[0, 0.1], pmap=optunity.pmap)
    
    print("Optimal parameters: " + str(optimal_rbf_pars))
    print("AUROC of tuned SVM with RBF kernel: %1.3f" % info.optimum)

.. parsed-literal::

    Optimal parameters: {'C': 5.145039160286679, 'gamma': 0.0011649329771152538}
    AUROC of tuned SVM with RBF kernel: 0.985


Tune SVC without deciding the kernel in advance 
------------------------------------------------

In the previous part we choose to use an RBF kernel. Even though the RBF
kernel is known to work well for a large variety of problems (and
yielded good accuracy here), our choice was somewhat arbitrary.

We will now use Optunity's conditional hyperparameter optimization
feature to optimize over all kernel functions and their associated
hyperparameters at once. This requires us to define the search space.

.. code:: python

    space = {'kernel': {'linear': {'C': [0, 2]},
                        'rbf': {'gamma': [0, 0.1], 'C': [0, 10]},
                        'poly': {'degree': [2, 5], 'C': [0, 5], 'coef0': [0, 2]}
                        }
             }
We will also have to modify the objective function to cope with
conditional hyperparameters. The reason we need to do this explicitly is
because scikit-learn doesn't like dealing with ``None`` values for
irrelevant hyperparameters (e.g. ``degree`` when using an RBF kernel).
Optunity will set all irrelevant hyperparameters in a given set to
``None``.

.. code:: python

    def train_model(x_train, y_train, kernel, C, gamma, degree, coef0):
        """A generic SVM training function, with arguments based on the chosen kernel."""
        if kernel == 'linear':
            model = sklearn.svm.SVC(kernel=kernel, C=C)
        elif kernel == 'poly':
            model = sklearn.svm.SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
        elif kernel == 'rbf':
            model = sklearn.svm.SVC(kernel=kernel, C=C, gamma=gamma)
        else: 
            raise ArgumentError("Unknown kernel function: %s" % kernel)
        model.fit(x_train, y_train)
        return model
    
    def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='linear', C=0, gamma=0, degree=0, coef0=0):
        model = train_model(x_train, y_train, kernel, C, gamma, degree, coef0)
        decision_values = model.decision_function(x_test)
        return optunity.metrics.roc_auc(y_test, decision_values)
    
    svm_tuned_auroc = cv_decorator(svm_tuned_auroc)
Now we are ready to go and optimize both kernel function and associated
hyperparameters!

.. code:: python

    optimal_svm_pars, info, _ = optunity.maximize_structured(svm_tuned_auroc, space, num_evals=150)
    print("Optimal parameters" + str(optimal_svm_pars))
    print("AUROC of tuned SVM: %1.3f" % info.optimum)

.. parsed-literal::

    Optimal parameters{'kernel': 'rbf', 'C': 7.919921875, 'coef0': None, 'gamma': 0.00107421875, 'degree': None}
    AUROC of tuned SVM: 0.986

