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


"""A variety of useful function decorators for logging and more.

Main features in this module:

* :func:`logged`
* :func:`max_evals`

.. moduleauthor:: Marc Claesen
"""

import collections
import functools
import threading
import operator as op


try:
    import pandas
    _pandas_available = True
except:
    _pandas_available = False

# http://stackoverflow.com/a/28752007
def wraps(obj, attr_names=functools.WRAPPER_ASSIGNMENTS):
    """Safe version of , that can deal with missing attributes
    such as missing __module__ and __name__.
    """
    return functools.wraps(obj, assigned=(name for name in attr_names
                                          if hasattr(obj, name)))


class Args(object):
    """Class to model arguments to a function evaluation.
    Objects of this class are hashable and can be used as dict keys.

    Arguments and keyword arguments are stored in a frozenset.
    """

    def __init__(self, *args, **kwargs):
        d = kwargs.copy()
        d.update(dict([('pos_' + str(i), item)
                       for i, item in enumerate(args)]))
        self._parameters = frozenset(sorted(d.items()))

    @property
    def parameters(self):
        """Returns the internal representation."""
        return self._parameters

    def __hash__(self):
        return hash(self.parameters)

    def __eq__(self, other):
        return (self.parameters) == (other.parameters)

    def __iter__(self):
        for x in self.parameters:
            yield x

    def __str__(self):
        return "{" + ", ".join(['\'' + str(k) + '\'' + ': ' + str(v)
                                for k, v in sorted(self.parameters)]) + "}"

    def _asdict(self):
        return dict([(k, v) for k, v in self.parameters])

    def keys(self):
        """Returns a list of argument names."""
        return map(op.itemgetter(0), self.parameters)

    def values(self):
        """Returns a list of argument values."""
        return map(op.itemgetter(1), self.parameters)


class CallLog(object):
    """Thread-safe call log.

    The call log is an ordered dictionary containing all previous function calls.
    Its keys are dictionaries representing the arguments and its values are the
    function values. As dictionaries can't be used as keys in dictionaries,
    a custom internal representation is used.
    """

    def __init__(self):
        """Initialize an empty CallLog."""
        self._data =collections.OrderedDict()
        self._lock = threading.Lock()

    @property
    def lock(self):
        return self._lock

    @property
    def data(self):
        """Access internal data after obtaining lock."""
        with self.lock:
            return self._data

    def delete(self, *args, **kwargs):
        del self.data[Args(*args, **kwargs)]

    def get(self, *args, **kwargs):
        """Returns the result of given evaluation or None if not previously done."""
        return self.data.get(Args(*args, **kwargs), None)

    def __setitem__(self, key, value):
        """Sets key=value in internal dictionary.

        :param key: key in the internal dictionary
        :type key: Args
        :param value: value in the internal dictionary
        :type value: float
        """
        assert(type(key) is Args)
        self.data[key] = value

    def __getitem__(self, key):
        """Returns the value corresponding to key. Can throw KeyError.

        :param key: arguments to retrieve function value for
        :type key: Args
        """
        assert(type(key) is Args)
        return self.data[key]

    def insert(self, value, *args, **kwargs):
        self.data[Args(*args, **kwargs)] = value

    def __iter__(self):
        for k, v in self.data:
            yield (dict([(key, val) for key, val in k]), v)

    def __len__(self):
        return len(self.data)

    def __nonzero__(self):
        return bool(self.data)

    def __str__(self):
        return "\n".join([str(k) + ' --> ' + str(v)
                          for k, v in self.data.items()])

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def update(self, other):
        assert(type(other) is CallLog)
        self.data.update(other.data)

    @staticmethod
    def from_dict(d):
        """Converts given dict to a valid call log used by logged functions.

        Given dictionary must have the following structure:
        ``{'args': {'argname': []}, 'values': []}``

        >>> log = CallLog.from_dict({'args': {'x': [1, 2]}, 'values': [2, 3]})
        >>> print(log)
        {'x': 1} --> 2
        {'x': 2} --> 3

        """
        log = CallLog()
        keys = d['args'].keys()
        for k, v in zip(zip(*d['args'].values()), d['values']):
            args = dict([(key, val) for key, val in zip(keys, k)])
            log.insert(v, **args)
        return log

    def to_dict(self):
        """Returns given call_log into a dictionary.

        The result is a dict with the following structure:
        ``{'args': {'argname': []}, 'values': []}``

        >>> call_log = CallLog()
        >>> call_log.insert(3, x=1, y=2)
        >>> d = call_log.to_dict()
        >>> d['args']['x']
        [1]
        >>> d['args']['y']
        [2]
        >>> d['values']
        [3]

        """
        if self.data:
            args = dict([(k, []) for k in list(self.keys())[0].keys()])
            values = []
            for k, v in self.data.items():
                for key, value in k:
                    args[key].append(value)
                values.append(v)
            return {'args': args, 'values': values}
        else:
            return {'args': {}, 'values': []}


def logged(f):
    """Decorator that logs unique calls to ``f``.

    The call log can always be retrieved using ``f.call_log``.
    Decorating a function that is already being logged has
    no effect.

    The call log is an instance of CallLog.

    >>> @logged
    ... def f(x): return x+1
    >>> a, b, c = f(1), f(1), f(2)
    >>> print(f.call_log)
    {'pos_0': 1} --> 2
    {'pos_0': 2} --> 3

    logged as inner decorator:

    >>> from .constraints import constrained
    >>> @logged
    ... @constrained([lambda x: x > 1])
    ... def f2(x): return x+1
    >>> len(f2.call_log)
    0
    >>> f2(2)
    3
    >>> print(f2.call_log)
    {'pos_0': 2} --> 3

    logged as outer decorator:

    >>> from .constraints import constrained
    >>> @constrained([lambda x: x > 1])
    ... @logged
    ... def f3(x): return x+1
    >>> len(f3.call_log)
    0
    >>> f3(2)
    3
    >>> print(f3.call_log)
    {'pos_0': 2} --> 3

    >>> @logged
    ... def f(x): return 1
    >>> f(1)
    1
    >>> print(f.call_log)
    {'pos_0': 1} --> 1
    >>> @logged
    ... @wraps(f)
    ... def f2(x): return f(x)
    >>> print(f2.call_log)
    {'pos_0': 1} --> 1

    """
    if hasattr(f, 'call_log'):
        return f

    @wraps(f)
    def wrapped_f(*args, **kwargs):
        value = wrapped_f.call_log.get(*args, **kwargs)
        if value is None:
            value = f(*args, **kwargs)
            wrapped_f.call_log.insert(value, *args, **kwargs)
        return value
    wrapped_f.call_log = CallLog()
    return wrapped_f


def negated(f):
    """Decorator to negate f such that f'(x) = -f(x)."""
    @wraps(f)
    def wrapped_f(*args, **kwargs):
        return -f(*args, **kwargs)
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
    the maximum is reached. Adds a field ``f.num_evals`` which tracks
    the number of evaluations that have been performed.

    >>> @max_evals(1)
    ... def f(x): return 2
    >>> f(2)
    2
    >>> f(1) #doctest:+SKIP
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
        @wraps(f)
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
    >>> f(3, 5)
    11

    """
    def wrapper(f):
        @wraps(f)
        def wrapped_f(*args):
            return f(**dict([(k, v) for k, v in zip(keys, args)]))
        return wrapped_f
    return wrapper


def call_log2dataframe(log):
    """Converts a call log into a pandas data frame.
    This function errors if you don't have pandas available.

    :param log: call log to be converted, as returned by e.g. `optunity.minimize`
    :returns: a pandas data frame capturing the same information as the call log

    """
    if not _pandas_available:
        raise NotImplementedError('This function requires pandas')

    args = log['args']
    values = log['values']
    hpar_names = args.keys()

    # construct a list of dictionaries
    zipped= zip(zip(*args.values()), values)
    dictlist = [dict([(k, v) for k, v in zip(hpar_names, args)] + [('value', value)])
                for args, value in zipped]
    df = pandas.DataFrame(dictlist)
    return df


if __name__ == '__main__':
    pass
