Grid Search
============

.. include:: /global.rst

This solver is implemented in |api-gridsearch|. It as available in |make_solver| as 'grid search'.


Grid search is an undirected search method that consists of testing a predefined set of values per hyperparameter. A search grid is constructed that is 
the Cartesian product of these sets of values.

Grid search can be used to tune a limited number of hyperparameters, as the size of the search grid (number of evaluations) increases exponentially in terms of the
number of hyperparameters. We generally recommend the use of directed solvers over grid search, as they waste less time exploring uninteresting regions.

When you use choose this solver in |maximize| or |minimize|, an equal number of values is determined per hyperparameter (spread uniformly).
The number of test values per hyperparameter is determined based on the number of permitted evaluations. 
If you want to specify the grid points manually, this is possible via |make_solver| and |optimize|.

Example
--------

Assume we have two hyperparameters `x` and `y`. For `x` we want to test the following values `[0, 1, 2, 3]` and for `y` we will try `[-10, -5, 0, 5, 10]`.
The search grid consists of all possible pairs (:math:`4\times5=20` in this case), e.g.:

+---+-----+
| x |  y  |
+===+=====+
| 0 | -10 |
+---+-----+
| 0 | -5  | 
+---+-----+
| 0 | 0   | 
+---+-----+
| 0 | 5   | 
+---+-----+
| 0 | 10  | 
+---+-----+
| 1 | -10 |
+---+-----+
| 1 | -5  | 
+---+-----+
| 1 | 0   | 
+---+-----+
| 1 | 5   | 
+---+-----+
| 1 | 10  | 
+---+-----+
| 2 | -10 |
+---+-----+
| 2 | -5  | 
+---+-----+
| 2 | 0   | 
+---+-----+
| 2 | 5   | 
+---+-----+
| 2 | 10  | 
+---+-----+
| 3 | -10 |
+---+-----+
| 3 | -5  | 
+---+-----+
| 3 | 0   | 
+---+-----+
| 3 | 5   | 
+---+-----+
| 3 | 10  | 
+---+-----+
