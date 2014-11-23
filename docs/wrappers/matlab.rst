MATLAB
=======

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

.. manual_:

Manual
-------

To obtain the manual of all solvers and a list of available solver names, use `optunity.manual()`.
If you specify a solver name, its manual will be printed out.

.. optimizing_:

Optimizing hyperparameters
---------------------------

The MATLAB wrapper offers `optunity.maximize()`, `optunity.minimize()` and `optunity.optimize()`. These
provide the same functionality as their Python equivalents.

Differences between Python and MATLAB version of `optimize`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MATLAB version has an extra argument `constraints` where you can specify domain constraints 
(the MATLAB wrapper has no equivalent of :func:`optunity.wrap_constraints`). Constraints are
communicated as a struct with the correct field names and corresponding values (more info at |constraints|).

As an example, to add constraints :math:`x < 3` and `y \geq 0`, we use the following code::

    constraints = struct('ub_o', struct('x', 3), 'lb_c', struct('y', 0));

Information about the solvers can be obtained at |solvers|. To learn about the specific parameter names
for each solver, check out the |api-solvers|.

.. cross-validation_:

Cross-validation
-----------------
