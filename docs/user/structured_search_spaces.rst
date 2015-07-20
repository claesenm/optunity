===================
Structured search spaces
===================

Some hyperparameter optimization problems have a hierarchical nature, comprising discrete choices and depending on the choice additional hyperparameters may exist.
A common example is optimizing a kernel, without choosing a certain family of kernel functions in advance (e.g. polynomial, RBF, linear, ...).

Optunity provides the functions :func:`optunity.maximize_structured` and :func:`optunity.minimize_structured` for such structured search spaces. 
Structured search spaces can be specified as nested dictionaries, which generalize the standard way of specifying box constraints:

- hyperparameters within box constraints: specified as dictionary entries, where `key=parameter name` and `value=box constraints (list)`.
- discrete choices: specified as a dictionary, where each entry represents a choice, that is `key=option name` and `value` has two options
    - a new dictionary of conditional hyperparameters, following the same rules
    - `None`, to indicate a choice which doesn't imply further hyperparameterization

Structured search spaces can be nested to form any graph-like search space. It's worth noting that the addition of discrete choices naturally generalizes Optunity's search space definition in :func:`optunity.minimize` and :func:`optunity.maximize`,
since box constraints are specified as keyword arguments there, so Python's `kwargs` to these functions effectively follows the exact same structure, e.g.:

.. code::

    _ = optunity.minimize(fun, num_evals=1, A=[0, 1], B=[-1, 2])
    # kwargs = {A: [0, 1], B: [-1, 2]}

Example: SVM kernel hyperparameterization
------------------------------------------

Suppose we want to optimize the kernel, choosing from the following options:

* linear kernel :math:`\kappa_{linear}(u, v) = u^T v`: no additional hyperparameters
* polynomial kernel :math:`\kappa_{poly}(u, v) = (u^T v + coef0)^{degree}`: 2 hyperparameters (degree and coef0)
* RBF kernel :math:`\kappa_{RBF}(u, v) = exp(-\gamma * |u-v|^2)` 1 hyperparameter (gamma)

When we put this all together, the SVM kernel search space can be defined as follows (Python syntax):

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

