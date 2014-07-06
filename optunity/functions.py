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

import itertools
import collections
import functools


def constr_ub_o(field, bounds, *args, **kwargs):
    """Models args.field < bounds."""
    return kwargs[field] < bounds


def constr_ub_c(field, bounds, *args, **kwargs):
    """Models args.field <= bounds."""
    return kwargs[field] <= bounds


def constr_lb_o(field, bounds, *args, **kwargs):
    """Models args.field > bounds."""
    return kwargs[field] > bounds


def constr_lb_c(field, bounds, *args, **kwargs):
    """Models args.field >= bounds."""
    return kwargs[field] >= bounds


def constr_range_oo(field, bounds, *args, **kwargs):
    """Models args.field in (bounds[0], bounds[1])."""
    return kwargs[field] > bounds[0] and kwargs[field] < bounds[1]


def constr_range_cc(field, bounds, *args, **kwargs):
    """Models args.field in [bounds[0], bounds[1]]."""
    return kwargs[field] >= bounds[0] and kwargs[field] <= bounds[1]


def constr_range_oc(field, bounds, *args, **kwargs):
    """Models args.field in (bounds[0], bounds[1]]."""
    return kwargs[field] > bounds[0] and kwargs[field] <= bounds[1]


def constr_range_co(field, bounds, **kwargs):
    """Models args.field in [bounds[0], bounds[1])."""
    return kwargs[field] >= bounds[0] and kwargs[field] < bounds[1]


class ConstraintViolation(Exception):
    """Thrown when constraints are not met."""
    def __init__(self, constraint, *args, **kwargs):
        self.__constraint = constraint
        self.__args = args
        self.__kwargs = kwargs

    @property
    def args(self):
        return self.__args

    @property
    def constraint(self):
        return self.__constraint

    @property
    def kwargs(self):
        return self.__kwargs


def constrained(constraints):
    """Decorator that puts constraints on the domain of f.

    >>> @constrained([lambda x: x > 0])
    ... def f(x): return x+1
    >>> f(1)
    2
    >>> f(0)
    Traceback (most recent call last):
    ...
    ConstraintViolation
    >>> len(f.constraints)
    1

    """
    def wrapper(f):
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            violations = [c for c in wrapped_f.constraints
                          if not c(*args, **kwargs)]
            if violations:
                raise ConstraintViolation(violations, *args, **kwargs)
            return f(*args, **kwargs)
        wrapped_f.constraints = constraints
        return wrapped_f
    return wrapper


def violations_defaulted(default):
    """Decorator to default function value when a ConstraintViolation occurs.

    >>> @violations_defaulted("foobar")
    ... @constrained([lambda x: x > 0])
    ... def f(x): return x+1
    >>> f(1)
    2
    >>> f(0)
    'foobar'

    """
    def wrapper(f):
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except ConstraintViolation:
                return default
        return wrapped_f
    return wrapper


def logged(f):
    """Decorator that logs unique calls to f.

    The call log can always be retrieved using f.call_log.
    Decorating a function that is already being logged has
    no effect.

    The call log is an OrderedDict with a namedtuple as key.
    The namedtuple has fields based on *args and **kwargs,
    for *args the tuple has fields pos_<i>.

    >>> @logged
    ... def f(x): return x+1
    >>> a, b, c = f(1), f(1), f(2)
    >>> f.call_log
    OrderedDict([(args(pos_0=1), 2), (args(pos_0=2), 3)])

    logged as inner decorator:
    >>> @logged
    ... @constrained([lambda x: x > 1])
    ... def f2(x): return x+1
    >>> f2.call_log
    OrderedDict()
    >>> f2(2)
    3
    >>> f2.call_log
    OrderedDict([(args(pos_0=2), 3)])

    logged as outer decorator:
    >>> @constrained([lambda x: x > 1])
    ... @logged
    ... def f3(x): return x+1
    >>> f3.call_log
    OrderedDict()
    >>> f3(2)
    3
    >>> f3.call_log
    OrderedDict([(args(pos_0=2), 3)])

    logging twice does not remove original call_log
    >>> @logged
    ... def f(x): return 1
    >>> f(1)
    1
    >>> f.call_log
    OrderedDict([(args(pos_0=1), 1)])
    >>> @logged
    ... @functools.wraps(f)
    ... def f2(x): return f(x)
    >>> f2.call_log
    OrderedDict([(args(pos_0=1), 1)])

    """
    if hasattr(f, 'call_log'):
        return f

    @functools.wraps(f)
    def wrapped_f(*args, **kwargs):
        d = kwargs.copy()
        d.update(dict([('pos_' + str(i), item)
                       for i, item in enumerate(args)]))
        if not wrapped_f.argtuple:
            f.argtuple = collections.namedtuple('args', d.keys())
        t = f.argtuple(**d)
        value = wrapped_f.call_log.get(t)
        if value is None:
            value = f(*args, **kwargs)
            wrapped_f.call_log[t] = value
        return value
    wrapped_f.call_log = collections.OrderedDict()
    wrapped_f.argtuple = None
    return wrapped_f


def dict2call_log(calldict):
    """Converts given dict to a valid call log used by logged functions.

    Given dictionary must have the following structure:
    {'args': {'argname': []}, 'values': []}

    >>> dict2call_log({'args': {'x': [1, 2]}, 'values': [2, 3]})
    OrderedDict([(Pars(x=1), 2), (Pars(x=2), 3)])

    """
    Pars = collections.namedtuple('Pars', calldict['args'].keys())
    return collections.OrderedDict((Pars(*args), val) for args, val in
                                   zip(zip(*calldict['args'].values()),
                                       calldict['values']))


def call_log2dict(call_log):
    """Returns given call_log into a dictionary.

    The call_log is an OrderedDict((namedtuple, value)).
    The result is a dict with the following structure:
    {'args': {'argname': []}, 'values': []}

    >>> Pars = collections.namedtuple('Pars',['x','y'])
    >>> call_log = collections.OrderedDict({Pars(1,2): 3})
    >>> call_log2dict(call_log)
    {'args': {'y': [2], 'x': [1]}, 'values': [3]}

    """
    if call_log:
        args = dict([(k, [getattr(arg, k) for arg in call_log.keys()])
                     for k in list(call_log.keys())[0]._fields])
            # note: wrap keys() in list to bypass view in Python 3
        return {'args': args, 'values': list(call_log.values())}
        # again: wrap in list() or JSON serializing fails in Python 3
    else:
        return {'args': {}, 'values': []}


def negated(f):
    """Decorator to negate f such that f'(x) = -f(x)."""
    @functools.wraps(f)
    def wrapped_f(*args, **kwargs):
        return -f(*args, **kwargs)
    return wrapped_f


def sequenced(f):
    """Decorator which sequences function evaluations when given a list
    of argument lists."""
    @functools.wraps(f)
    def wrapped_f(*args, **kwargs):  # FIXME: how to handle this?
        if islist(args):
            return [f(arg) for arg in args]
        else:
            return f(args)
    return wrapped_f


class MaximumEvaluationsException(Exception):
    """Raised when the maximum number of function evaluations are used."""
    def __init__(self, max_evals):
        self._max_evals = max_evals

    @property
    def max_evals(self):
        """Returns the maximum number of evaluations that was permitted."""
        return self._max_evals


def max_evals(max_evals):
    """Decorator to enforce a maximum number of function evaluations.

    Throws a MaximumEvaluationsException during evaluations after
    the maximum is reached. Adds a field f.num_evals which tracks
    the number of evaluations that have been performed.

    >>> @max_evals(1)
    ... def f(x): return 2
    >>> f(2)
    2
    >>> f(1)
    Traceback (most recent call last):
    ...
    MaximumEvaluationsException
    >>> try:
    ...    f(1)
    ... except MaximumEvaluationsException as e:
    ...    e.max_evals
    1

    """
    def wrapper(f):
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            if wrapped_f.num_evals >= max_evals:
                raise MaximumEvaluationsException(max_evals)
            else:
                wrapped_f.num_evals += 1
                return f(*args, **kwargs)
        wrapped_f.num_evals = 0
        return wrapped_f
    return wrapper


def static_key_order(keys):
    """Decorator to fix the key order for use in function evaluations.

    A fixed key order allows the function to be evaluated with a list of
    unnamed arguments rather than kwargs.

    >>> @static_key_order(['foo', 'bar'])
    ... def f(bar, foo): return bar + 2 * foo
    >>> f([3,5])
    11

    """
    def wrapper(f):
        @functools.wraps(f)
        def wrapped_f(args):
            return f(**dict([(k, v) for k, v in zip(keys, args)]))
        return wrapped_f
    return wrapper
