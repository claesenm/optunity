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
* :func:`suggest_solver`
* :func:`manual`
* :func:`maximize`
* :func:`minimize`
* :func:`optimize`

We recommend using these functions rather than equivalents found in other places,
e.g. :mod:`optunity.solvers`.

.. moduleauthor:: Marc Claesen

"""

import functools
import timeit
import sys
import operator
import collections

# optunity imports
from . import functions as fun
from . import solvers
from solvers import solver_registry
from .util import DocumentedNamedTuple as DocTup


def _manual_lines(solver_name=None):
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


def manual(solver_name=None):
    """Prints the manual of requested solver.

    :param solver_name: (optional) name of the solver to request a manual from.
        If none is specified, a general manual is printed.

    Raises ``KeyError`` if ``solver_name`` is not registered."""
    if solver_name:
        man = solver_registry.get(solver_name).desc_full
    else:
        man = solver_registry._manual_lines()
    print('\n'.join(man))


optimize_results = DocTup("""
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
                          'optimize_results', ['optimum',
                                               'stats',
                                               'call_log',
                                               'report']
                          )
optimize_stats = DocTup("""
**Statistics gathered while solving a problem**:

num_evals
    number of function evaluations
time
    wall clock time needed to solve
                        """,
                        'optimize_stats', ['num_evals', 'time'])


def suggest_solver(num_evals=50, solver_name=None, **kwargs):
    if solver_name:
        solvercls = solver_registry.get(solver_name)
    else:
        solver_name = 'particle swarm'
        solvercls = solvers.ParticleSwarm
    if hasattr(solvercls, 'suggest_from_box'):
        suggestion = solvercls.suggest_from_box(num_evals, **kwargs)
    elif hasattr(solvercls, 'suggest_from_seed'):
        # the seed will be the center of the box that is provided to us
        seed = dict([(k, float(v[0] + v[1]) / 2) for k, v in kwargs.items()])
        suggestion = solvercls.suggest_from_seed(num_evals, **seed)
    else:
        raise ValueError('Unable to instantiate ' + solvercls.name + '.')
    suggestion['solver_name'] = solver_name
    return suggestion


def maximize(f, num_evals=50, solver_name=None, pmap=map, **kwargs):
    """Basic function maximization routine. Maximizes ``f`` within
    the given box constraints.

    :param f: the function to be maximized
    :param num_evals: number of permitted function evaluations
    :param solver_name: name of the solver to use (optional)
    :type solver_name: string
    :param pmap: the map function to use
    :type pmap: callable
    :param kwargs: box constraints, a dict of the following form
        ``{'parameter_name': [lower_bound, upper_bound], ...}``
    :returns: retrieved maximum, extra information and solver info

    This function will implicitly choose an appropriate solver and
    its initialization based on ``num_evals`` and the box constraints.

    """
    # sanity check on box constraints
    assert all([len(v) == 2 and v[0] < v[1]
                for v in kwargs.values()]), 'Box constraints improperly specified: should be [lb, ub] pairs'

    f = _wrap_hard_box_constraints(f, kwargs, sys.float_info.min)

    suggestion = suggest_solver(num_evals, solver_name, **kwargs)
    solver = make_solver(**suggestion)
    solution, details = optimize(solver, f, maximize=True, max_evals=num_evals,
                                 pmap=pmap)
    return solution, details, suggestion


def minimize(f, num_evals=50, solver_name=None, pmap=map, **kwargs):
    """Basic function minimization routine. Minimizes ``f`` within
    the given box constraints.

    :param f: the function to be minimized
    :param num_evals: number of permitted function evaluations
    :param solver_name: name of the solver to use (optional)
    :type solver_name: string
    :param pmap: the map function to use
    :type pmap: callable
    :param kwargs: box constraints, a dict of the following form
        ``{'parameter_name': [lower_bound, upper_bound], ...}``
    :returns: retrieved minimum, extra information and solver info

    This function will implicitly choose an appropriate solver and
    its initialization based on ``num_evals`` and the box constraints.

    """
    # sanity check on box constraints
    assert all([len(v) == 2 and v[0] < v[1]
                for v in kwargs.values()]), 'Box constraints improperly specified: should be [lb, ub] pairs'

    func =  _wrap_hard_box_constraints(f, kwargs, sys.float_info.max)

    suggestion = suggest_solver(num_evals, solver_name, **kwargs)
    solver = make_solver(**suggestion)
    solution, details = optimize(solver, func, maximize=False, max_evals=num_evals,
                                 pmap=pmap)
    return solution, details, suggestion


def optimize(solver, func, maximize=True, max_evals=0, pmap=map):
    """Optimizes func with given solver.

    Returns the solution and a namedtuple with further details.
    Please refer to docs of optunity.maximize_results
    and optunity.maximize_stats.

    Raises KeyError if
    - ``solver_name`` is not registered
    - ``solver_config`` is invalid to instantiate ``solver_name``

    """

    if max_evals > 0:
        f = fun.max_evals(max_evals)(func)
    else:
        f = func

    f = fun.logged(f)
    num_evals = -len(f.call_log)

    time = timeit.default_timer()
    try:
        solution, report = solver.optimize(f, maximize, pmap=pmap)
    except fun.MaximumEvaluationsException:
        # early stopping because maximum number of evaluations is reached
        # retrieve solution from the call log
        report = None
        if maximize:
            index, _ = max(enumerate(f.call_log.values()), key=operator.itemgetter(1))
        else:
            index, _ = min(enumerate(f.call_log.values()), key=operator.itemgetter(1))
        solution = operator.itemgetter(index)(f.call_log.keys())._asdict()
    time = timeit.default_timer() - time

    # FIXME: logged function's argtuple type remains None
    t = collections.namedtuple('args', f.keys)
#    optimum = f(**solution)
    optimum = f.call_log[t(**solution)]
#    optimum = f.call_log[solution]
    num_evals += len(f.call_log)

    # use namedtuple to enforce uniformity in case of changes
    stats = optimize_stats(num_evals, time)

    call_dict = fun.call_log2dict(f.call_log)
    return solution, optimize_results(optimum, stats._asdict(),
                                      call_dict, report)


optimize.__doc__ = '''
Optimizes func with given solver.

Returns the solution and a ``namedtuple`` with further details.
''' + optimize_results.__doc__ + optimize_stats.__doc__


def make_solver(solver_name, *args, **kwargs):
    """Creates a Solver from given parameters.

    :param solver_name: the solver to instantiate
    :type solver_name: string
    :param args: positional arguments to solver constructor.
    :param kwargs: keyword arguments to solver constructor.

    Use :func:`optunity.manual` to get a list of registered solvers.
    For constructor arguments per solver, please refer to :doc:`/user/solvers`.

    Raises ``KeyError`` if

    - ``solver_name`` is not registered
    - ``*args`` and ``**kwargs`` are invalid to instantiate the solver.

    """
    solvercls = solver_registry.get(solver_name)
    return solvercls(*args, **kwargs)


def wrap_call_log(f, call_dict):
    """Wraps an existing call log (as dictionary) around f.

    This allows you to communicate known function values to solvers.
    (currently available solvers do not use this info)

    """
    f = fun.logged(f)
    call_log = fun.dict2call_log(call_dict)
    if f.call_log:
        f.call_log.update(call_log)
    else:
        f.call_log = call_log
    return f


def wrap_constraints(f, default=None, ub_o=None, ub_c=None,
                     lb_o=None, lb_c=None, range_oo=None,
                     range_co=None, range_oc=None, range_cc=None):
    """Decorates f with given input domain constraints.

    :param f: the function that will be constrained
    :type f: callable
    :param default: function value to default to in case of constraint violations
    :type default: number
    :param ub_o: open upper bound constraints, e.g. :math:`x < c`
    :type ub_o: dict
    :param ub_c: closed upper bound constraints, e.g. :math:`x \leq c`
    :type ub_c: dict
    :param lb_o: open lower bound constraints, e.g. :math:`x > c`
    :type lb_o: dict
    :param lb_c: closed lower bound constraints, e.g. :math:`x \geq c`
    :type lb_c: dict
    :param range_oo: range constraints (open lb and open ub)
        :math:`lb < x < ub`
    :type range_oo: dict with 2-element lists as values ([lb, ub])
    :param range_co: range constraints (closed lb and open ub)
        :math:`lb \leq x < ub`
    :type range_co: dict with 2-element lists as values ([lb, ub])
    :param range_oc: range constraints (open lb and closed ub)
        :math:`lb < x \leq ub`
    :type range_oc: dict with 2-element lists as values ([lb, ub])
    :param range_cc: range constraints (closed lb and closed ub)
        :math:`lb \leq x \leq ub`
    :type range_cc: dict with 2-element lists as values ([lb, ub])

    >>> def f(x):
    ...     return x
    >>> fc = wrap_constraints(f, default=-1, range_cc={'x': [0, 1]})
    >>> fc(x=5)
    -1

    """
    kwargs = locals()
    del kwargs['f']
    del kwargs['default']
    for k, v in kwargs.items():
        if v is None:
            del kwargs[k]

    if not kwargs:
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
    for constr_name, pars in kwargs.items():
        constr_fun = jt[constr_name]
        for field, bounds in pars.items():
            constraints.append(functools.partial(constr_fun,
                                                 field,
                                                 bounds))

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


def _wrap_hard_box_constraints(f, box, default):
    """Places hard box constraints on the domain of ``f``
    and defaults function values if constraints are violated.

    :param f: the function to be wrapped with constraints
    :type f: callable
    :param box: the box, as a dict: ``{'param_name': [lb, ub], ...}``
    :type box: dict
    :param default: function value to default to when constraints
        are violated
    :type default: number

    """
    return wrap_constraints(f, default, range_oo=box)
