"""
Optunity
==========

Provides
  1. Routines to efficiently optimize hyperparameters
  2. Function decorators to implicitly log evaluations, constrain the domain and more.
  3. Facilities for k-fold cross-validation to estimate generalization performance.

Available modules
---------------------
solvers
    contains all officially supported solvers
functions
    a variety of useful function decorators for hyperparameter tuning
cross_validation
    k-fold cross-validation

Available subpackages
---------------------
test
    regression test suite

Utilities
---------

__version__
    Optunity version string
__revision__
    Optunity revision string
__author__
    Main authors of the package

"""

from .api import manual, print_manual, maximize, minimize, optimize
from .api import wrap_call_log, wrap_constraints, make_solver, suggest_solver
from .cross_validation import cross_validated, generate_folds

__author__ = "Marc Claesen, Jaak Simm and Dusan Popovic"
__version__ = "0.2"
__revision__ = "0.2.1"

__all__ = ['manual', 'print_manual', 'maximize', 'minimize', 'optimize',
           'wrap_call_log', 'wrap_constraints', 'make_solver',
           'suggest_solver', 'cross_validated', 'generate_folds']
