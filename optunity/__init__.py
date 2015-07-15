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
tests
    regression test suite
solvers
    solver implementations and auxillary functions

Utilities
---------

__version__
    Optunity version string
__revision__
    Optunity revision string
__author__
    Main authors of the package

"""

from .api import manual, maximize, minimize, optimize, available_solvers, maximize_structured, minimize_structured
from .api import wrap_call_log, wrap_constraints, make_solver, suggest_solver
from .cross_validation import cross_validated, generate_folds
from .parallel import pmap
from .functions import call_log2dataframe

__author__ = "Marc Claesen, Jaak Simm and Dusan Popovic"
__version__ = "1.0.0"
__revision__ = "1.0.1"

__all__ = ['manual', 'maximize', 'minimize', 'optimize',
           'wrap_call_log', 'wrap_constraints', 'make_solver',
           'suggest_solver', 'cross_validated', 'generate_folds',
           'pmap', 'available_solvers', 'call_log2dataframe',
           'maximize_structured', 'minimize_structured']
