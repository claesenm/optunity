MATLAB
=======

.. include:: /global.rst

In this page we briefly discuss the MATLAB wrapper, which provides most of Optunity's functionality. 
For a general overview, we recommend reading the :doc:`/user/index`.

For installation instructions, please refer to |installation|.

Whenever the Python API requires a dictionary, we use a MATLAB `struct`. As MATLAB has no keyword
arguments, we use `varargs` with the convention of `<name>`, `<value>`.

For MATLAB, the following main features are provided:

-   :ref:`manual`

-   :ref:`optimizing`

-   :ref:`cross-validation`

The file `<optunity>/wrappers/matlab/optunity_example.m` contains code for all the functionality that is
available in MATLAB.

.. note::

    If you experience any issues with the MATLAB wrapper, create a global variable `DEBUG_OPTUNITY`
    and set its value to true. This will show all debugging output. Please submit this output
    as an issue at |issues| to obtain help::

        global DEBUG_OPTUNITY
        DEBUG_OPTUNITY=true;

    If MATLAB hangs while using Optunity there is a communication issue. This should not occur, 
    if you encounter this please file an |issue|. Currently the only way out of this is to kill MATLAB.

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
This will return an `optunity.Solver` object or return an error message.

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
