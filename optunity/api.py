#! /usr/bin/env python

# Author: Marc Claesen
#
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


import functools

# optunity imports
from . import functions as fun
from . import solvers
from . import solver_registry


def manual_request(solver_name=None):
    """Returns the manual and name that was requested via solver.
    If no solver_name is specified, returns a list of all registered solvers.
    If a name was specified, returns the same name in a list.

    Raises KeyError if solver_name is not registered."""
    if solver_name:
        return solver_registry.get(solver_name).desc_full, [solver_name]
    else:
        return solver_registry.manual(), solver_registry.solver_names()


def maximize(solver, func):
    """Maximizes func with given solver.

    Returns the following:
        - solution: optimal argument tuple
        - optimum: f(solution)
        - num_evals: number of evaluations of f performed during maximization
        - call_log: record of all historical function evaluations of f
            returned as dict {'args': {'argname': []}, 'values': []}
            note: len(call_log) >= num_evals
        - solver report, can be None

    Raises KeyError if
        - <solver_name> is not registered
        - <solver_config> is invalid to instantiate <solver_name>
    """
    @fun.logged
    @functools.wraps(func)
    def f(*args, **kwargs):
        return func(*args, **kwargs)

    num_evals = -len(f.call_log)

    solution, report = solver.maximize(f)
    optimum = f(**solution)
    num_evals += len(f.call_log)

    call_dict = fun.call_log2dict(f.call_log)
    return solution, optimum, num_evals, call_dict, report


def make_solver(solver_name, solver_config):
    """Creates a Solver from given parameters."""
    solvercls = solver_registry.get(solver_name)
    return solvercls(**solver_config)


def wrap_call_log(f, call_dict):
    """Wraps an existing call log (as dictionary) around f."""
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
        - 'ub_?': upper bound
        - 'lb_?': lower bound
        - 'range_??': range'
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
