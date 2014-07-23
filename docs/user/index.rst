===================
Optunity User Guide
===================

This page contains a high-level overview of Optunity. 
The main module provides the basic user functionality. For more advanced
use, please refer to the submodules. Everything discussed on this page is available
in all Optunity wrappers, though specialized submodule features might not be. We
will now elaborate on the basic approach of a general tuning task:

BLAH

Optunity provides a variety of solvers for hyperparameter tuning problems.
A tuning problem is specified by an objective function which does the work
under the hood and provides a score for some tuple of hyperparameters. As
this is entirely dependent on user preferences, you must implement this yourself.
As an example, consider a support vector machine (SVM): the objective function
to test a specific parameter tuple must do the following:

1. Train an SVM model with given hyperparameters.
2. Test the SVM model on an independent test set.
3. Compute some score measure for the test predictions, for example accuracy.

Optunity offers the following main functions:

- :func:`optunity.maximize` 
- :func:`optunity.minimize` 
- :func:`optunity.optimize` 

.. toctree::
    :maxdepth: 2

    /user/installation
    /user/solvers
    /user/cross_validation
    /user/functions
