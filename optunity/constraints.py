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

"""
All functionality related to domain constraints on objective function.

Main features in this module:

* :func:`constrained`
* :func:`wrap_constraints`

.. moduleauthor:: Marc Claesen
"""

import functools
from . import functions

def constr_ub_o(field, bounds, *args, **kwargs):
    """Models ``args.field < bounds``."""
    return kwargs[field] < bounds


def constr_ub_c(field, bounds, *args, **kwargs):
    """Models ``args.field <= bounds``."""
    return kwargs[field] <= bounds


def constr_lb_o(field, bounds, *args, **kwargs):
    """Models ``args.field > bounds``."""
    return kwargs[field] > bounds


def constr_lb_c(field, bounds, *args, **kwargs):
    """Models ``args.field >= bounds``."""
    return kwargs[field] >= bounds


def constr_range_oo(field, bounds, *args, **kwargs):
    """Models ``args.field in (bounds[0], bounds[1])``."""
    return kwargs[field] > bounds[0] and kwargs[field] < bounds[1]


def constr_range_cc(field, bounds, *args, **kwargs):
    """Models ``args.field in [bounds[0], bounds[1]]``."""
    return kwargs[field] >= bounds[0] and kwargs[field] <= bounds[1]


def constr_range_oc(field, bounds, *args, **kwargs):
    """Models ``args.field in (bounds[0], bounds[1]]``."""
    return kwargs[field] > bounds[0] and kwargs[field] <= bounds[1]


def constr_range_co(field, bounds, **kwargs):
    """Models ``args.field in [bounds[0], bounds[1])``."""
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
    >>> f(0) #doctest:+SKIP
    Traceback (most recent call last):
    ...
    ConstraintViolation
    >>> len(f.constraints)
    1

    """
    def wrapper(f):
        @functions.wraps(f)
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
    """Decorator to default function value when a :class:`ConstraintViolation` occurs.

    >>> @violations_defaulted("foobar")
    ... @constrained([lambda x: x > 0])
    ... def f(x): return x+1
    >>> f(1)
    2
    >>> f(0)
    'foobar'

    """
    def wrapper(f):
        @functions.wraps(f)
        def wrapped_f(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except ConstraintViolation:
                return default
        return wrapped_f
    return wrapper


def wrap_constraints(f, default=None, ub_o=None, ub_c=None,
                     lb_o=None, lb_c=None, range_oo=None,
                     range_co=None, range_oc=None, range_cc=None,
                     custom=None):
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
    :param custom: custom, user-defined constraints
    :type custom: list of constraints

    *custom constraints are binary functions that yield False in case of violations.

    >>> def f(x):
    ...     return x
    >>> fc = wrap_constraints(f, default=-1, range_oc={'x': [0, 1]})
    >>> fc(x=0.5)
    0.5
    >>> fc(x=1)
    1
    >>> fc(x=5)
    -1
    >>> fc(x=0)
    -1

    We can define any custom constraint that we want. For instance,
    assume we have a binary function with arguments `x` and `y`, and we want
    to make sure that the provided values remain within the unit circle.

    >>> def f(x, y):
    ...     return x + y
    >>> circle_constraint = lambda x, y: (x ** 2 + y ** 2) <= 1
    >>> fc = wrap_constraints(f, default=1234, custom=[circle_constraint])
    >>> fc(0.0, 0.0)
    0.0
    >>> fc(1.0, 0.0)
    1.0
    >>> fc(0.5, 0.5)
    1.0
    >>> fc(1, 0.5)
    1234

    """
    kwargs = locals()
    del kwargs['f']
    del kwargs['default']
    del kwargs['custom']
    for k, v in list(kwargs.items()):
        if v is None:
            del kwargs[k]

    if not kwargs and not custom:
        return f

    # jump table to get the right constraint function
    jt = {'ub_o': constr_ub_o,
          'ub_c': constr_ub_c,
          'lb_o': constr_lb_o,
          'lb_c': constr_lb_c,
          'range_oo': constr_range_oo,
          'range_oc': constr_range_oc,
          'range_co': constr_range_co,
          'range_cc': constr_range_cc}

    # construct constraint list
    constraints = []
    for constr_name, pars in kwargs.items():
        constr_fun = jt[constr_name]
        for field, bounds in pars.items():
            constraints.append(functools.partial(constr_fun,
                                                 field=field,
                                                 bounds=bounds))
    if custom:
        constraints.extend(custom)

    # wrap function
    if default is None:
        @constrained(constraints)
        @functions.wraps(f)
        def func(*args, **kwargs):
            return f(*args, **kwargs)
    else:
        @violations_defaulted(default)
        @constrained(constraints)
        @functions.wraps(f)
        def func(*args, **kwargs):
            return f(*args, **kwargs)
    return func

