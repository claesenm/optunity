===================
Optunity User Guide
===================

This page contains a high-level overview of Optunity. 
The main module provides the basic user functionality. For more advanced
use, please refer to the submodules. Everything discussed on this page is available
in all Optunity wrappers, though specialized submodule features might not be. 

Some basic terminology we will use consistently throughout the documentation:

- *hyperparameters*: user-specified parameters for a given machine learning approach.
  These will serve as optimization variables in our context.
  Example: kernel parameters.
- *score*: some measure to quantify the quality of a certain model.
  Example: accuracy of a classifier.
- *objective function*: the function that must be optimized. The arguments to this function
  are a tuple of hyperparameters. 
  Example: accuracy of an SVM classifier with certain kernel and regularization parameters.
- *solver*: a strategy to optimize hyperparameters.
  Example: grid search.
- *train-predict-score (TPS) chain*: 

Optunity provides a variety of solvers for hyperparameter tuning problems.
A tuning problem is specified by an objective function which does the work
under the hood and provides a score for some tuple of hyperparameters. As
this is entirely dependent on user preferences, you must implement this yourself.
As an example, consider a support vector machine (SVM): the objective function
to test a specific parameter tuple must do the following:

1. Train an SVM model with given hyperparameters.
2. Test the SVM model on an independent test set.
3. Compute some score measure for the test predictions, for example accuracy.

Optunity consists of a set of core functions that are offered in each environment,
which we will now discuss briefly. Clicking on a function will take you to its Python
API documentation. If you are using a different environment, you can still get the
general idea on the Python pages. Optunity offers the following core functions:

- :func:`optunity.maximize` 
- :func:`optunity.minimize` 
- :func:`optunity.optimize` 
- :func:`optunity.suggest_solver` 
- :func:`optunity.make_solver`
- :func:`optunity.manual`
- :func:`optunity.cross_validated`



.. toctree::
    :maxdepth: 2

    /user/installation
    /user/solvers
    /user/cross_validation
    /user/functions
