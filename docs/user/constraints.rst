Domain constraints
===================

.. include:: /global.rst

Optunity supports domain constraints on the objective function. Domain constraints are used to enforce solvers to remain within a prespecified search space.
Most solvers that Optunity provides are implicitly unconstrained (cfr. |solvers|), though hyperparameters are usually constrained in some way (ex: regularization coefficients must be positive).

A set of simple constraints and facilities to use them are provided in |api-constraints|. Specifically, the following constraints are provided:

-   `lb_{oc}`: assigns a lower bound (open or closed)
-   `ub_{oc}`: assigns an upper bound (open or closed)
-   `range_{oc}{oc}`: creates a range constraint (e.g. :math:`A < x < B`)

All of the above constraints apply to a specific hyperparameter. Multidimensional constraints are possible, 
but you would need to implement them yourself (see :ref:`custom-constraints`).

Note that the functions |maximize| and |minimize| wrap explicit box constraints around the objective function prior to starting the solving process. The expert function
|optimize| does not do this for you, which allows more flexibility at the price of verbosity.

Constraint violations in Optunity raise a `ConstraintViolation` exception by default. 
The usual way we handle these exceptions is by returning a certain (typically bad) default function value (using the :func:`optunity.constraints.violations_defaulted` decorator).
This will cause solvers to stop searching in the infeasible region.

To add a series of constraints, we recommend using the |wrap-constraints| function. 
This function takes care of assigning default values on constraint violations if desired.

.. _custom-constraints:

Implementing custom constraints
-------------------------------

Constraints are implemented as a binary functions, which yield false in case of a constraint violation. You can design your own constraint according to this principle.
For instance, assume we have a binary function with two arguments `x` and `y`::

    def f(x, y):
        return x + y

Optunity provides all univariate constraints you need, but lets say we want to constrain the domain to be the unit circle in `x,y`-space. 
We can do this using the following constraint::

    constraint = lambda x, y: (x ** 2 + y ** 2) <= 1.0

To constrain `f`, we use |wrap-constraints|::

    fc = optunity.wrap_constraints(f, custom=[constraint])

The constrained function `fc(x, y)` will yield `x + y` if the arguments are within the unit circle, or raise a `ConstraintViolation` exception otherwise.
