#! /usr/bin/env python

# Copyright (c) 2014 KU Leuven, ESAT-STADIUS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither name of copyright holders nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""A collection of top-level API functions for Optunity.

Main functions in this module:

* :func:`make_solver`
* :func:`manual`
* :func:`maximize`
* :func:`tune`

We recommend using these functions rather than equivalents found in other places,
e.g. :mod:`optunity.solvers`.

.. moduleauthor:: Marc Claesen

"""

import functools
import timeit

# optunity imports
from . import functions as fun
from . import solvers
from . import solver_registry
from .util import DocumentedNamedTuple as DocTup


def manual(solver_name=None):
    """Brief solver manual.

    :param solver_name: (optional) name of the solver to request a manual from.
        If none is specified, a general manual and list of all registered solvers is returned.

    :result:
        * list of strings that contain the requested manual
        * solver name(s): name of the solver that was specified or list of all registered solvers.

    Raises ``KeyError`` if ``solver_name`` is not registered."""
    if solver_name:
        return solver_registry.get(solver_name).desc_full, [solver_name]
    else:
        return solver_registry.manual(), solver_registry.solver_names()


def print_manual(solver_name=None):
    """Prints the manual of requested solver.

    :param solver_name: (optional) name of the solver to request a manual from.
        If none is specified, a general manual is printed.

    Raises ``KeyError`` if ``solver_name`` is not registered."""
    if solver_name:
        man = solver_registry.get(solver_name).desc_full
    else:
        man = solver_registry.manual()
    print('\n'.join(man))


maximize_results = DocTup("""
**Result details includes the following**:

optimum
    optimal function value f(solution)

stats
    statistics about the solving process

call_log
    the call log

report
    solver report, can be None
                          """,
                          'maximize_results', ['optimum',
                                               'stats',
                                               'call_log',  'report'])
maximize_stats = DocTup("""
**Statistics gathered while solving a problem**:

num_evals
    number of function evaluations
time
    wall clock time needed to solve
                        """,
                        'maximize_stats', ['num_evals', 'time'])


def maximize(solver, func):
    """Maximizes func with given solver.

    Returns the solution and a namedtuple with further details.
    Please refer to docs of optunity.maximize_results
    and optunity.maximize_stats.

    Raises KeyError if
    - ``solver_name`` is not registered
    - ``solver_config`` is invalid to instantiate ``solver_name``

    """
    @fun.logged
    @functools.wraps(func)
    def f(*args, **kwargs):
        return func(*args, **kwargs)

    num_evals = -len(f.call_log)

    time = timeit.default_timer()
    solution, report = solver.maximize(f)
    time = timeit.default_timer() - time

    optimum = f(**solution)
    num_evals += len(f.call_log)

    # use namedtuple to enforce uniformity in case of changes
    stats = maximize_stats(num_evals, time)

    call_dict = fun.call_log2dict(f.call_log)
    return solution, maximize_results(optimum, stats._asdict(),
                                      call_dict, report)


maximize.__doc__ = '''
Maximizes func with given solver.

Returns the solution and a ``namedtuple`` with further details.
''' + maximize_results.__doc__ + maximize_stats.__doc__


def tune(f, num_evals, solver_name=None, *args, **kwargs):
    if solver_name:
        solvercls = solver_registry.get(solver_name)
    else:
        solvercls = solver_registry.get('particle swarm')
    pass  # TODO: where to implement this logic?


def make_solver(solver_name, *args, **kwargs):
    """Creates a Solver from given parameters.

    Raises ``KeyError`` if

    - ``solver_name`` is not registered
    - ``*args`` and ``**kwargs`` are invalid to instantiate the solver.

    """
    solvercls = solver_registry.get(solver_name)
    return solvercls(*args, **kwargs)


def wrap_call_log(f, call_dict):
    """Wraps an existing call log (as dictionary) around f.

    """
    f = fun.logged(f)
    call_log = fun.dict2call_log(call_dict)
    if f.call_log:
        f.call_log.update(call_log)
    else:
        f.call_log = call_log
    return f


def wrap_constraints(f, constraint_dict, default=None):
    """Decorates f with all constraints listed in the dict.

    constraint_dict may have the following keys:

    - ``ub_?``: upper bound
    - ``lb_?``: lower bound
    - ``range_??``: range'

    where '?' can be either 'o' (open) or 'c' (closed).
    The values of constraint_dict are dicts with argname-value pairs.

    """
    if not constraint_dict:
        return f

    # jump table to get the right constraint function
    jt = {'ub_o': fun.constr_ub_o,
          'ub_c': fun.constr_ub_c,
          'lb_o': fun.constr_lb_o,
          'lb_c': fun.constr_lb_c,
          'range_oo': fun.constr_range_oo,
          'range_oc': fun.constr_range_oc,
          'range_co': fun.constr_range_co,
          'range_cc': fun.constr_range_cc}

    # construct constraint list
    constraints = []
    for constr_name, pars in constraint_dict.items():
        constr_fun = jt[constr_name]
        for field, bounds in pars.items():
            constraints.append(functools.partial(constr_fun,
                                                 field=field,
                                                 bounds=bounds))

    # wrap function
    if default is None:
        @fun.constrained(constraints)
        @functools.wraps(f)
        def func(*args, **kwargs):
            return f(*args, **kwargs)
    else:
        @fun.violations_defaulted(default)
        @fun.constrained(constraints)
        @functools.wraps(f)
        def func(*args, **kwargs):
            return f(*args, **kwargs)
    return func
