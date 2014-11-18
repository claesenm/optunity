===================
User Guide
===================

This page contains a high-level overview of Optunity. 
The main module provides the basic user functionality. For more advanced
use, please refer to the submodules. Everything discussed on this page is available
in all Optunity wrappers, though specialized submodule features might not be. 

If you want to learn by example, please consult our :doc:`/examples/index`, 
which use various features of Optunity to cover a wide variety of tuning tasks. 
In case of confusion, we provide a list of basic `Terminology`_.

Introduction
-------------

Optunity provides a variety of solvers for hyperparameter tuning problems.
A tuning problem is specified by an objective function that provides a score for 
some tuple of hyperparameters. Specifying the objective function must be done by
the user. The software offers a diverse set of solvers to optimize the objective
function. A solver determines an optimal tuple of hyperparameters.

In a more mathematical sense, tuning is about maximizing the following objective function:

.. math::

    \lambda^* = arg\,min_{\lambda} \mathcal{L}\big(\mathcal{A}(\mathbf{X}^{(tr)}\ |\ \lambda)\ |\ \mathbf{X}^{(te)}\big) = arg\,min_{\lambda} \mathcal{F}(\lambda\ |\ \mathcal{A},\ \mathbf{X}^{(tr)}, \mathbf{X}^{(tr)},\ \mathcal{L}) :label: objective-function


.. sidebar:: **Jump to**

    * :doc:`Solver docs </user/solvers>`
    * :doc:`Cross-validation </user/cross_validation>`

Optunity consists of a set of core functions that are offered in each environment,
which we will now discuss briefly. Clicking on a function will take you to its Python
API documentation. If you are using a different environment, you can still get the
general idea on the Python pages. To dive into code details straight away, please
consult the :doc:`/api/optunity`.

In the rest of this section we will discuss the main API functions. We will start with
very simple functions that offer basic functionality which should meet the needs of most
use cases. Subsequently we will introduce the expert interface functions which have more
bells and whistles that can be configured.

A variety of solvers is available, but they are not discussed on this page. 
For more details about our solvers, please visit :doc:`/user/solvers`. 

Simple interface
----------------

For beginning users, we offer a set of functions with simple arguments. These functions
should be enough for most of your needs. In case these functions are insufficient, 
please refer to the expert functions listed below or to submodules.

- :func:`optunity.maximize`: maximizes the objective function
    Adheres to a prespecified upper bound on the number of function evaluations.
    The solution will be within given box constraints. Optunity determines
    the best solver and its configuration for you.
- :func:`optunity.minimize`: minimizes the objective function
    Adheres to a prespecified upper bound on the number of function evaluations.
    The solution will be within given box constraints. Optunity determines
    the best solver and its configuration for you.
- :func:`optunity.manual`: prints a basic manual (general or solver specific)
    Prints a basic manual of Optunity and a list of all registered solvers.
    If a solver name is specified, its manual will be shown.

    .. note::
        You may alternatively consult the solver API documentation at 
        :doc:`/api/optunity.solvers` for more details.

- :func:`optunity.cross_validated`: decorator to perform k-fold cross-validation
    Wrap a function with cross-validation functionality. The way cross-validation
    is performed is highly configurable, including support for strata and clusters.

Expert interface
-----------------

The following functions are recommended for more advanced use of Optunity. This
part of the API allows you to fully configure every detail about the provided solvers.
In addition to more control in configuration, you can also perform parallel function
evaluations via this interface (turned off by default due to problems in IPython).

- :func:`optunity.suggest_solver`: suggests a solver and its configuration
    Suggests a solver and configuration for a given tuning problem, based
    on the permitted number of function evaluations and box constraints.
- :func:`optunity.make_solver`: constructs one of Optunity's registered solvers.
    See the solver-specific manuals for more information per solver.
- :func:`optunity.optimize`: optimizes an objective function with given solver
    Some solvers are capable of vector evaluations. By default, the optimization
    is done through sequential function evaluations but this can be parallelized
    by specifying an appropriate ``pmap`` argument (cfr. :func:`optunity.parallel.pmap`).


Terminology
------------

To avoid confusion, here is some basic terminology we will use consistently 
throughout the documentation:

.. glossary::

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

    box constraints
        Every hyperparameter of the tuning problem must be within a prespecified interval. 
        The optimal solution will be within the hyperrectangle (box) specified by the ranges.

    solver
        A strategy to optimize hyperparameters, such as *grid search*.

    train-predict-score (TPS) chain
        A sequence of code which trains a model, uses it to predict an independent test set and 
        then computes some score measure based on the predictions. TPS chains must be specified 
        by the user as they depend entirely on the method that is being tuned and the evaluation criteria.

.. toctree::
    :maxdepth: 2

    /user/installation
    /user/solvers
    /user/cross_validation
    /user/decorators
