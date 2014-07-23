===================
Optunity User Guide
===================

This page contains a high-level overview of Optunity. 
The main module provides the basic user functionality. For more advanced
use, please refer to the submodules. Everything discussed on this page is available
in all Optunity wrappers, though specialized submodule features might not be. 

Terminology
------------

To avoid confusion, we will first introduce some basic terminology we will use consistently 
throughout the documentation:

hyperparameters
    User-specified parameters for a given machine learning approach.
    These will serve as optimization variables in our context.

    Example: kernel parameters.

score
    Some measure which quantifies the quality of a certain modeling approach (model type + hyperparameters).

    Example: cross-validated accuracy of a classifier.

objective function
    The function that must be optimized. The arguments to this function are hyperparameters
    and the result is some score measure of a model constructed using these hyperparameters.
    Optunity can minimize and maximize, depending on the requirements.
    
    Example: accuracy of an SVM classifier as a function of kernel and regularization parameters.

solver
    A strategy to optimize hyperparameters, such as *grid search*.

train-predict-score (TPS) chain
    A sequence of code which trains a model, uses it to predict an independent test set and 
    then computes some score measure based on the predictions. TPS chains must be specified 
    by the user as they depend entirely on the method that is being tuned and the evaluation criteria.


Introduction
-------------

Optunity provides a variety of solvers for hyperparameter tuning problems.
A tuning problem is specified by an objective function that provides a score for 
some tuple of hyperparameters. Specifying the objective function must be done by
the user. The software offers a diverse set of solvers to optimize the objective
function. A solver determines an optimal tuple of hyperparameters.

Optunity consists of a set of core functions that are offered in each environment,
which we will now discuss briefly. Clicking on a function will take you to its Python
API documentation. If you are using a different environment, you can still get the
general idea on the Python pages. Optunity offers the following core functions:

- :func:`optunity.maximize`: maximizes the objective function with a prespecified 
    number of function evaluations with certain box constraints. Optunity determines
    the best solver and its configuration for you.
- :func:`optunity.minimize`: minimizes the objective function with a prespecified 
    number of function evaluations with certain box constraints. Optunity determines
    the best solver and its configuration for you.
- :func:`optunity.optimize` 
- :func:`optunity.suggest_solver` 
- :func:`optunity.make_solver`
- :func:`optunity.manual`
- :func:`optunity.cross_validated`


A variety of solvers is available. For more details, please visit :doc:`/user/solvers`.

.. toctree::
    :maxdepth: 2

    /user/installation
    /user/solvers
    /user/cross_validation
    /user/functions
