Nelder-Mead simplex
===================

.. include:: /global.rst

This solver is implemented in |api-nelder-mead|. It is available in |make_solver| as 'nelder-mead'.

This is a heuristic, nonlinear optimization method based on the concept of a simplex, originally introduced by Nelder and Mead [NELDERMEAD]_.
We implemented the version as described on Wikipedia_.

This method requires an initial starting point :math:`x_0`. It is a good local search method, but will get stuck in bad regions when a poor starting point is specified.


.. [NELDERMEAD] Nelder, John A. and Mead, R. *A simplex method for function minimization*. Computer Journal 7: 308–313, 1965.

.. _Wikipedia: http://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
