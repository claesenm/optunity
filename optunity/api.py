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
* :func:`maximize_structured`
* :func:`minimize`
* :func:`minimize_structured`
* :func:`optimize`

We recommend using these functions rather than equivalents found in other places,
e.g. :mod:`optunity.solvers`.

.. moduleauthor:: Marc Claesen

"""

import timeit
import sys
import operator

# optunity imports
from . import functions as fun
from . import solvers
from . import search_spaces
from .solvers import solver_registry
from .util import DocumentedNamedTuple as DocTup
from .constraints import wrap_constraints


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


def available_solvers():
    """Returns a list of all available solvers.

    These can be used in :func:`optunity.make_solver`.
    """
    return solver_registry.solver_names()


def manual(solver_name=None):
    """Prints the manual of requested solver.

    :param solver_name: (optional) name of the solver to request a manual from.
        If none is specified, a general manual is printed.

    Raises ``KeyError`` if ``solver_name`` is not registered."""
    if solver_name:
        man = solver_registry.get(solver_name).desc_full
    else:
        man = solver_registry.manual()
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

    f = _wrap_hard_box_constraints(f, kwargs, -sys.float_info.max)

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


def optimize(solver, func, maximize=True, max_evals=0, pmap=map, decoder=None):
    """Optimizes func with given solver.

    :param solver: the solver to be used, for instance a result from :func:`optunity.make_solver`
    :param func: the objective function
    :type func: callable
    :param maximize: maximize or minimize?
    :type maximize: bool
    :param max_evals: maximum number of permitted function evaluations
    :type max_evals: int
    :param pmap: the map() function to use, to vectorize use :func:`optunity.parallel.pmap`
    :type pmap: function

    Returns the solution and a namedtuple with further details.
    Please refer to docs of optunity.maximize_results
    and optunity.maximize_stats.

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
        solution = list(f.call_log.keys())[index]._asdict()
    time = timeit.default_timer() - time

    # TODO why is this necessary?
    if decoder: solution = decoder(solution)

    optimum = f.call_log.get(**solution)
    num_evals += len(f.call_log)

    # use namedtuple to enforce uniformity in case of changes
    stats = optimize_stats(num_evals, time)

    call_dict = f.call_log.to_dict()
    return solution, optimize_results(optimum, stats._asdict(),
                                      call_dict, report)


optimize.__doc__ = '''
Optimizes func with given solver.

:param solver: the solver to be used, for instance a result from :func:`optunity.make_solver`
:param func: the objective function
:type func: callable
:param maximize: maximize or minimize?
:type maximize: bool
:param max_evals: maximum number of permitted function evaluations
:type max_evals: int
:param pmap: the map() function to use, to vectorize use :func:`optunity.pmap`
:type pmap: function

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
    call_log = fun.CallLog.from_dict(call_dict)
    if f.call_log:
        f.call_log.update(call_log)
    else:
        f.call_log = call_log
    return f


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


def maximize_structured(f, search_space, num_evals=50, pmap=map):
    """Basic function maximization routine. Maximizes ``f`` within
    the given box constraints.

    :param f: the function to be maximized
    :param search_space: the search space (see :doc:`/user/structured_search_spaces` for details)
    :param num_evals: number of permitted function evaluations
    :param pmap: the map function to use
    :type pmap: callable
    :returns: retrieved maximum, extra information and solver info

    This function will implicitly choose an appropriate solver and
    its initialization based on ``num_evals`` and the box constraints.

    """
    tree = search_spaces.SearchTree(search_space)
    box = tree.to_box()

    # we need to position the call log here
    # because the function signature used later on is internal logic
    f = fun.logged(f)

    # wrap the decoder and constraints for the internal search space representation
    f = tree.wrap_decoder(f)
    f = _wrap_hard_box_constraints(f, box, -sys.float_info.max)

    suggestion = suggest_solver(num_evals, "particle swarm", **box)
    solver = make_solver(**suggestion)
    solution, details = optimize(solver, f, maximize=True, max_evals=num_evals,
                                 pmap=pmap, decoder=tree.decode)
    return solution, details, suggestion

def minimize_structured(f, search_space, num_evals=50, pmap=map):
    """Basic function minimization routine. Minimizes ``f`` within
    the given box constraints.

    :param f: the function to be maximized
    :param search_space: the search space (see :doc:`/user/structured_search_spaces` for details)
    :param num_evals: number of permitted function evaluations
    :param pmap: the map function to use
    :type pmap: callable
    :returns: retrieved maximum, extra information and solver info

    This function will implicitly choose an appropriate solver and
    its initialization based on ``num_evals`` and the box constraints.

    """
    tree = search_spaces.SearchTree(search_space)
    box = tree.to_box()

    # we need to position the call log here
    # because the function signature used later on is internal logic
    f = fun.logged(f)

    # wrap the decoder and constraints for the internal search space representation
    f = tree.wrap_decoder(f)
    f = _wrap_hard_box_constraints(f, box, sys.float_info.max)

    suggestion = suggest_solver(num_evals, "particle swarm", **box)
    solver = make_solver(**suggestion)
    solution, details = optimize(solver, f, maximize=False, max_evals=num_evals,
                                 pmap=pmap, decoder=tree.decode)
    return solution, details, suggestion

