===================
Structured search spaces
===================

Some hyperparameter optimization problems have a hierarchical nature, comprising discrete choices and depending on the choice additional hyperparameters may exist.
A common example is optimizing a kernel, without choosing a certain family of kernel functions in advance (e.g. polynomial, RBF, linear, ...).

Optunity provides the functions :func:`optunity.maximize_structured` and :func:`optunity.minimize_structured` for such structured search spaces. 
Suppose we want to optimize the kernel, choosing from the following options:

* linear kernel :math:`\kappa_{linear}(u, v) = u^T v`: no additional hyperparameters
* polynomial kernel :math:`\kappa_{poly}(u, v) = (u^T v + coef0)^{degree}`: 2 hyperparameters (degree and coef0)
* RBF kernel :math:`\kappa_{RBF}(u, v) = exp(-\gamma * |u-v|^2)` 1 hyperparameter (gamma)

To optimize a structured search space, we have to define it first, which can be defined as follows (Python syntax):

.. code::

    search = {'kernel': {'linear': None,
                         'rbf': {'gamma': [gamma_min, gamma_max]},
                         'poly': {'degree': [degree_min, degree_max],
                                  'coef0': [coef0_min, coef0_max]}
                         }
               }

The structure of the search space directly matches the hyperparameterization of every kernel function. 
We use `None` in the linear kernel as there are no additional hyperparameters. The hyperparameters of the RBF and polynomial kernel follow
Optunity's default syntax based on dictionnaries.

Optunity also supports nested choices, for example an outer choice for the learning algorithm (e.g. SVM, naive Bayes, ...) and an inner choice for the SVM kernel function.
This is illustrated in the following notebook: :doc:`/notebooks/notebooks/sklearn-automated-classification`.

