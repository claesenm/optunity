Domain constraints
===================

.. include:: /global.rst

Optunity supports domain constraints on the objective function. Domain constraints are used to enforce solvers to remain within a prespecified search space.
Most solvers that Optunity provides are implicitly unconstrained (cfr. |solvers|), though hyperparameters are obviously constrained in some way (ex: regularization coefficients must be positive).

A set of simple constraints and facilities to use them are provided in :module:`optunity.functions`. Specifically, the following constraints are provided:

-   `lb_{oc}`: assigns a lower bound (open or closed)
-   `ub_{oc}`: assigns an upper bound (open or closed)
-   `range_{oc}{oc}`: creates a range constraint (e.g. :math:`A < x < B`)

All of the above constraints apply to a specific hyperparameter. Multidimensional constraints are possible, 
but you would need to implement them yourself (see :ref:`custom-constraints`).

Note that the functions |maximize| and |minimize| wrap explicit box constraints around the objective function prior to starting the solving process. The expert function
|optimize| does not do this for you, which allows more flexibility at the price of verbosity.

To add your own set of constraints, we recommend using the |wrap-constraints| function.

.. _custom-constraints:

Implementing custom constraints
-------------------------------

Constraints are implemented as a binary functions, which yield false in case of a constraint violation. You can design your own constraint according to this principle.

maximize and minimize do it for you

make your own and attach with wrap_constraints()
