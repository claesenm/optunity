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

import abc


class MetaDocumentedSolver(abc.ABCMeta):
    """Provides class properties related to solver documentation."""
    @property
    def name(cls):
        """Returns the name of this Solver."""
        return cls._name

    @property
    def desc_brief(cls):
        """Returns a one-line description of this Solver."""
        return cls._desc_brief

    @property
    def desc_full(cls):
        """Returns a full description of this Solver, including manual."""
        return cls._desc_full


# python version-independent metaclass usage
DocumentedSolver = MetaDocumentedSolver('DocumentedSolver', (object, ), {})

__registered_solvers = {}


def __register(cls):
    global __registered_solvers
    __registered_solvers[cls.name] = cls


def get(solver_name):
    """Returns the class of given solver."""
    global __registered_solvers
    return __registered_solvers[solver_name]


def manual():
    global __registered_solvers
    manual = ['Optunity: optimization algorithms for hyperparameter tuning', ' ',
                'The following solvers are available:']
    for name, cls in __registered_solvers.items():
        manual.append(name + ' :: ' + cls.desc_brief)
    manual.append(' ')
    manual.append("For a solver-specific manual, include its name in the request.")
    manual.append("For more detailed info, please consult the Optunity user wiki at:")
    manual.append("https://github.com/claesenm/optunity/wiki/User-homepage")
    return manual


def solver_names():
    global __registered_solvers
    return list(__registered_solvers.keys())


def register_solver(name, desc_brief, desc_full):
    """Class decorator to register solver in the registry.

    Attaches attributes to cls before registering:
        - _name attribute indicating solver name
        - _desc_full attribute with extensive description and manual
        - _desc_brief attribute with one-line description

    These attributes are available as class properties.
    """
    def class_wrapper(cls):
        class wrapped_solver(DocumentedSolver, cls):
            _name = name
            _desc_brief = desc_brief
            _desc_full = desc_full

            __name__ = cls.__name__
            __doc__ = cls.__doc__
            __module__ = cls.__module__

            def __init__(self, *args, **kwargs):
                super(wrapped_solver, self).__init__(*args, **kwargs)

        # register the new wrapped_solver class
        __register(wrapped_solver)
        return wrapped_solver
    return class_wrapper
