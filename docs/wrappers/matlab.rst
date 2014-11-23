MATLAB
=======

.. include:: /global.rst

In this page we briefly discuss the MATLAB wrapper, which provides most of Optunity's functionality. 
For a general overview, we recommend reading the :doc:`/user/index`.

For installation instructions, please refer to |installation|.

Whenever the Python API requires a dictionary, we use a MATLAB `struct`. As MATLAB has no keyword
arguments, we use `varargs` with the convention of `<name>`, `<value>`. Please refer to each
function's help for details on how to use it.

.. warning:: Since MATLAB does not have keyword arguments, every objective function you define
    should accept a single struct argument, whose fields represent hyperparameters.

    Example, to model a function :math:`f(x, y) = x + y`::

        f = @(pars) pars.x + pars.y;


For MATLAB, the following main features are provided:

-   :ref:`manual`

-   :ref:`optimizing`

-   :ref:`cross-validation`

The file `<optunity>/wrappers/matlab/optunity_example.m` contains code for all the functionality that is
available in MATLAB.

.. note::

    If you experience any issues with the MATLAB wrapper, create a global variable `DEBUG_OPTUNITY`
    and set its value to true. This will show all debugging output::

        global DEBUG_OPTUNITY
        DEBUG_OPTUNITY=true;

    Please submit this output as an issue at |new-issue| to obtain help.

    If MATLAB hangs while using Optunity there is a communication issue. This should not occur, 
    if you encounter this please file an issue at |new-issue|. Currently the only way out of this is to kill MATLAB.

.. _manual:

Manual
-------

To obtain the manual of all solvers and a list of available solver names, use `optunity.manual()`.
If you specify a solver name, its manual will be printed out.

You can verify whether or not a certain solver, for instance `cma-es`, is available like this::

    solvers = optunity.manual();
    available = any(arrayfun(@(x) strcmp(x, 'cma-es'), solvers));


.. _optimizing:

Optimizing hyperparameters
---------------------------

The MATLAB wrapper offers `optunity.maximize()`, `optunity.minimize()` and `optunity.optimize()`. These
provide the same functionality as their Python equivalents.

The following code fragment shows how to optimize a simple function `f` with |randomsearch| within the box
:math:`-4 < x < 4` and :math:`-5 < y < 5` and 200 evaluations::
    
    offx = rand();
    offy = rand();
    f = @(pars) - (offx+pars.x)^2 - (offy+pars.y)^2;

    [max_solution, max_details, max_solver] = optunity.maximize(f, 200, ...
            'solver_name', 'random search', 'x', [-4, 4], 'y', [-5, 5]);

If you want to use `optunity.optimize()`, you must create the solver in advance using `optunity.make_solver()`.
This will return an `optunity.Solver` object or return an error message::

    f = @(pars) pars.x + pars.y
    rnd_solver = optunity.make_solver('random search', 'x', [-5, 5], ...
            'y', [-5, 5], 'num_evals', 400);
    [rnd_solution, rnd_details] = optunity.optimize(rnd_solver, f);


Differences between Python and MATLAB version of `optimize`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MATLAB version has an extra argument `constraints` where you can specify domain constraints 
(the MATLAB wrapper has no equivalent of :func:`optunity.wrap_constraints`). Constraints are
communicated as a struct with the correct field names and corresponding values (more info at |constraints|).

As an example, to add constraints :math:`x < 3` and :math:`y \geq 0`, we use the following code::

    constraints = struct('ub_o', struct('x', 3), 'lb_c', struct('y', 0));

Information about the solvers can be obtained at |solvers|. To learn about the specific parameter names
for each solver, check out the |api-solvers|.

.. _cross-validation:

Cross-validation
-----------------

Two functions are provided for cross-validation:

-   `optunity.generate_folds()`: generates a set of cross-validation folds

-   `optunity.cross_validate()`: function decorator, to perform cross-validation with specified function

Both functions can deal with strata and clusters. You can specify these as a cell array of vectors of indices::

    strata = {[1,2,3], [6,7,8,9]};
    folds = optunity.generate_folds(20, 'num_folds', 10, 'num_iter', 2, 'strata', strata);

Cross-validation folds are returned as a matrix of `num_instances * num_iter` with entries ranging from 1 to `num_folds` to indicate
the fold each instance belongs to per iteration.

`optunity.cross_validate()` requires a function handle as its first argument. This is the function that will be decorated, which
must have the following first arguments: `x_train` and `x_test` (if unsupervised) or `x_train, y_train, x_test, y_test`.

As an example, assume we have a function `optunity_cv_fun(x_train, x_test, pars)`::

    function [ result ] = cv_fun( x_train, x_test, pars )
    disp('training set:');
    disp(x_train');
    disp('test set:');
    disp(x_test');

    result = -pars.x^2 - pars.y^2;
    end

This must be decorated with cross-validation, for instance::

    x = (1:10)';
    cvf = optunity.cross_validate(@optunity_cv_fun, x);

    % evaluate the function: this will return a cross-validation result
    performance = cvf(struct('x',1,'y',2));

.. warning:: 

    After decorating with cross-validation, the objective function should have a single argument,
    namely a struct of hyperparameters.

