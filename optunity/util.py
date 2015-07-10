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

import collections
import itertools
import inspect

def nth(iterable, n):
    """Returns the nth item from iterable."""
    try:
        return iterable[n]
    except TypeError:
        try:
            return next(itertools.islice(iterable, n, None))
        except StopIteration:
            raise IndexError('index out of range')


def DocumentedNamedTuple(docstring, *ntargs):
    """Factory function to construct collections.namedtuple with
    a docstring. Useful to attach meta-information to data structures.

    Inspired by http://stackoverflow.com/a/1606478"""
    nt = collections.namedtuple(*ntargs)

    class NT(nt):
        __doc__ = docstring
    return NT


def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    if not callable(func):
        raise TypeError("%s is not callable" % type(func))
    if inspect.isfunction(func):
        print('a')
        spec = inspect.getargspec(func)
    elif hasattr(func, 'im_func'):
        print('b')
        spec = inspect.getargspec(func.im_func)
        print(spec)
    elif inspect.isclass(func):
        print('c')
        return get_default_args(func.__init__)
    elif isinstance(func, object):
        print('d')
        # We already know the instance is callable,
        # so it must have a __call__ method defined.
        # Return the arguments it expects.
        return get_default_args(func.__call__)

    try:
        return dict(zip(reversed(spec.args), reversed(spec.defaults)))
    except TypeError:
        # list of defaults and/or args are empty
        return {}


# taken from http://kbyanc.blogspot.be/2007/07/python-more-generic-getargspec.html
def getargspec(obj):
    """Get the names and default values of a callable's
       arguments

    A tuple of four things is returned: (args, varargs,
    varkw, defaults).
      - args is a list of the argument names (it may
        contain nested lists).
      - varargs and varkw are the names of the * and
        ** arguments or None.
      - defaults is a tuple of default argument values
        or None if there are no default arguments; if
        this tuple has n elements, they correspond to
        the last n elements listed in args.

    Unlike inspect.getargspec(), can return argument
    specification for functions, methods, callable
    objects, and classes.  Does not support builtin
    functions or methods.
    """
    if not callable(obj):
        raise TypeError("%s is not callable" % type(obj))
    try:
        if inspect.isfunction(obj):
            return inspect.getargspec(obj)
        elif hasattr(obj, 'im_func'):
            # For methods or classmethods drop the first
            # argument from the returned list because
            # python supplies that automatically for us.
            # Note that this differs from what
            # inspect.getargspec() returns for methods.
            # NB: We use im_func so we work with
            #     instancemethod objects also.
            spec = list(inspect.getargspec(obj.im_func))
            spec[0] = spec[0][1:]
            return spec
        elif inspect.isclass(obj):
            return getargspec(obj.__init__)
        #elif isinstance(obj, object) and \
            #     not isinstance(obj, type(arglist.__get__)): # TODO: what does this clause do?
        elif isinstance(obj, object):
            # We already know the instance is callable,
            # so it must have a __call__ method defined.
            # Return the arguments it expects.
            return getargspec(obj.__call__)
    except NotImplementedError:
        # If a nested call to our own getargspec()
        # raises NotImplementedError, re-raise the
        # exception with the real object type to make
        # the error message more meaningful (the caller
        # only knows what they passed us; they shouldn't
        # care what aspect(s) of that object we actually
        # examined).
        pass
    raise NotImplementedError("do not know how to get argument list for %s" % type(obj))
