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

"""Module to take care of registering solvers for use in the main Optunity API.

Main functions in this module:

* :func:`register_solver`
* :func:`manual`
* :func:`get`

.. moduleauthor:: Marc Claesen

"""

__all__ = ['get', 'manual', 'register_solver', 'solver_names']

__registered_solvers = {}


def __register(cls):
    global __registered_solvers
    __registered_solvers[cls.name.lower()] = cls


def get(solver_name):
    """Returns the class of the solver registered under given name.

    :param solver_name: name of the solver
    :returns: the solver class"""
    global __registered_solvers
    return __registered_solvers[solver_name]


def manual():
    """
    Returns the general manual of Optunity, with a brief introduction of all registered solvers.

    :returns: the manual as a list of strings (lines)
    """
    global __registered_solvers
    manual = ['Optunity: optimization algorithms for hyperparameter tuning', ' ',
                'The following solvers are available:']
    for name, cls in __registered_solvers.items():
        manual.append(name + ' :: ' + cls.desc_brief)
    manual.append(' ')
    manual.append("For a solver-specific manual, include its name in the request.")
    manual.append("For more detailed info, please consult the Optunity documentation at:")
    manual.append("http://docs.optunity.net")
    return manual


def solver_names():
    """Returns a list of all registered solvers."""
    global __registered_solvers
    return list(__registered_solvers.keys())


def register_solver(name, desc_brief, desc_full):
    """Class decorator to register a :class:`optunity.solvers.Solver` subclass in the registry.
    Registered solvers will be available through Optunity's main API functions,
    e.g. :func:`optunity.make_solver` and :func:`optunity.manual`.

    :param name: name to register the solver with
    :param desc_brief: one-line description of the solver
    :param desc_full:
        extensive description and manual of the solver
        returns a list of strings representing manual lines
    :returns: a class decorator to register solvers in the solver registry

    The resulting class decorator attaches attributes to the class before registering:

    :_name: the name using which the solver is registered
    :_desc_full: extensive description and manual as list of strings (lines)
    :_desc_brief: attribute with one-line description

    These attributes will be available as class properties.
    """
    def class_wrapper(cls):
        cls.name = name
        cls.desc_brief = desc_brief
        cls.desc_full = desc_full
        __register(cls)
        return cls
    return class_wrapper
