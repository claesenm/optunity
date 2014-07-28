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

"""This module is for developer use only. It is used to start a
communication session.

.. warning::
    Importing this module should only be done when launching Optunity
    as a subprocess to access functionality from non-Python environments.
    Do not import this module as a Python user.


This module implements several use cases to provide the main Optunity API in
other environments. It communicates with the external environment using JSON messages.

We will discuss the use cases as if they are ordinary function calls. Parameter names
are keys in the JSON root message.

Requesting manuals
-------------------

Emulates :func:`optunity.manual`.

Request:

+--------+------------------------------------------------------------+----------+
| Key    | Value                                                      | Optional |
+========+============================================================+==========+
| manual | name of the solver whose manual we want or empty string    | no       |
+--------+------------------------------------------------------------+----------+

Reply:

+--------------+-----------------------------------------------+-----------------+
| Key          | Value                                         | Type            |
+==============+===============================================+=================+
| manual       | the manual that was requested                 | list of strings |
+--------------+-----------------------------------------------+-----------------+
| solver_names | the name of the solver whose manual was       | list of strings |
|              | requested or a list of all registered solvers |                 |
+--------------+-----------------------------------------------+-----------------+


Generate cross-validation folds
---------------------------------

Use the Python library to generate k-fold cross-validation folds. This requires one message back and forth.

Emulates :func:`optunity.generate_folds`.

Request:

+----------------+-----------------------------------------------------+------------+
| Key            | Value                                               | Optional   |
+================+=====================================================+============+
| generate_folds | dictionary:                                         | no         |
|                |                                                     |            |
|                | - **num_instances** number of instances to consider | - no       |
|                | - **num_folds** number of folds                     | - yes (10) |
|                | - **num_iter** number of iterations                 | - yes (1)  |
|                | - **strata** to account for in fold generation      | - yes      |
|                | - **clusters** to account for in fold generation    | - yes      |
+----------------+-----------------------------------------------------+------------+

Strata and clusters are sent as list (strata/clusters) of lists (instance indices).

.. note::
    strata and cluster indices must be 0-based

Reply:

+--------------+---------------------+------------------------------------+
| Key          | Value               | Type                               |
+==============+=====================+====================================+
| folds        | the resulting folds | list (iterations) of lists (folds) |
+--------------+---------------------+------------------------------------+

The inner lists contain the instance indices per fold (0-based indexing).

Maximize
---------

Emulates :func:`optunity.maximize`.

Using the simple maximize functionality involves sending an initial message with the
arguments to :func:`optunity.maximize` as shown below.

Subsequently, the solver will send objective function evaluation requests sequentially.
These requests can be for scalar or vector evaluations (details below).

When the solution is found, or the
maximum number of evaluations is reached a final message is sent with all details.

Setup request:

+-------------+----------------------------------------------------------+------------+
| Key         | Value                                                    | Optional   |
+=============+==========================================================+============+
| maximize    | dictionary:                                              | no         |
|             |                                                          |            |
|             | - **num_evals** number of permitted function evaluations | - no       |
|             | - **box constraints** dictionary                         | - no       |
+-------------+----------------------------------------------------------+------------+
| call_log    | a call log of previous function evaluations              | yes        |
+-------------+----------------------------------------------------------+------------+
| constraints | domain constraints on the objective function             | yes        |
|             |                                                          |            |
|             | - **ub_{oc}** upper bound (open/closed)                  | - yes      |
|             | - **lb_{oc}** lower bound (open/closed)                  | - yes      |
|             | - **range_{oc}{oc}** interval bounds                     | - yes      |
+-------------+----------------------------------------------------------+------------+

After the initial setup message, Optunity will send objective function evaluation requests.
These request may be for a scalar or a vector evaluation, and look like this:

**Scalar evaluation request**: the message is a dictionary containing the hyperparameter
names as keys and their associated values as values.

An example request to evaluate f(x, y) in (x=1, y=2):

.. highlight:: JSON

    {"x": 1, "y": 2}


**Vector evaluation request**: the message is a list of dictionaries with the same form as
the dictionary of a scalar evaluation request.

An example request to evaluate f(x, y) in (x=1, y=2) and (x=2, y=3):

.. highlight:: JSON

    [{"x": 1, "y": 2}, {"x": 2, "y": 3}]

The replies to evaluation requests are simple:

- scalar request: dictionary with key *value* and value the objective function value
- vector request: dictionary with key *values* and value a list of function values

.. note::

    Results of vector evaluations must be returned in the same order as the request.


When a solution is found, Optunity will send a final message with all necessary information
and then exit. This final message contains the following:




Minimize
---------

Identical to maximize (above) except that the initial message has the key
``minimize`` instead of ``maximize``.

Emulates :func:`optunity.minimize`.

Optimize
---------

Emulates :func:`optunity.optimize`.


Optimization related fields
----------------------------



.. moduleauthor:: Marc Claesen

"""

from __future__ import print_function
import sys

# optunity imports
from . import communication as comm
from . import functions
import optunity


def manual_request(solver_name):
    """Emulates :func:`optunity.manual`."""
    if len(solver_name) == 0:
        solver_name = None
    try:
        manual, solver_names = optunity.api._manual_lines(solver_name)
    except KeyError:
        msg = {'error_msg': 'Solver does not exist (' + solver_name + ').'}
        comm.send(comm.json_encode(msg))
        print(startup_msg, file=sys.stderr)
        exit(1)
    except EOFError:
        msg = {'error_msg': 'Broken pipe.'}
        comm.send(comm.json_encode(msg))
        exit(1)
    msg = {'manual': manual, 'solver_names': solver_names}
    comm.send(comm.json_encode(msg))
    exit(0)


def fold_request(cv_opts):
    """Computes k-fold cross-validation folds.
    Emulates :func:`optunity.cross_validated`."""
    try:
        num_instances = cv_opts['num_instances']
    except KeyError:
        msg = {'error_msg': 'number of instances num_instances must be set.'}
        comm.send(comm.json_encode(msg))
        print(startup_msg, file=sys.stderr)
        exit(1)

    num_folds = cv_opts.get('num_folds', 10)
    strata = cv_opts.get('strata', None)
    clusters = cv_opts.get('clusters', None)
    num_iter = cv_opts.get('num_iter', 1)

    idx2cluster = None
    if clusters:
        idx2cluster = cv.map_clusters(clusters)

    folds = [optunity.generate_folds(num_instances, num_folds=num_folds,
                                     strata=strata, clusters=clusters,
                                     idx2cluster=idx2cluster)
             for _ in range(num_iter)]

    msg = {'folds': folds}
    comm.send(comm.json_encode(msg))
    exit(0)


def prepare_fun(mgr, constraints, default, call_log):
    """Creates the objective function and wraps it with domain constraints
    and an existing call log, if applicable."""
    func = optunity.wrap_constraints(comm.make_piped_function(mgr),
                                     default, **constraints)
    if call_log:
        func = optunity.wrap_call_log(func, call_log)
    else:
        func = functions.logged(func)
    return func


def max_or_min(solve_fun, kwargs, constraints, default, call_log):
    """Emulates :func:`optunity.maximize` and :func:`optunity.minimize`."""
    # prepare objective function
    mgr = comm.EvalManager()
    func = prepare_fun(mgr, constraints, default, call_log)

    # solve problem
    try:
        solution, rslt, solver = solve_fun(func, pmap=mgr.pmap, **kwargs)
    except EOFError:
        msg = {'error_msg': 'Broken pipe.'}
        comm.send(comm.json_encode(msg))
        exit(1)

    # send solution and exit
    result = rslt._asdict()
    result['solution'] = solution
    result['solver'] = solver
    result_json = comm.json_encode(result)
    comm.send(result_json)
    exit(0)


def optimize(solver, constraints, default, call_log):
    """Emulates :func:`optunity.optimize`."""
    # prepare objective function
    mgr = comm.EvalManager()
    func = prepare_fun(mgr, constraints, default, call_log)

    # solve problem
    try:
        solution, rslt, solver = solve_fun(func, pmap=mgr.pmap, **kwargs)
    except EOFError:
        msg = {'error_msg': 'Broken pipe.'}
        comm.send(comm.json_encode(msg))
        exit(1)

    # send solution and exit
    result = rslt._asdict()
    result['solution'] = solution
    result['solver'] = solver
    result_json = comm.json_encode(result)
    comm.send(result_json)
    exit(0)


if __name__ == '__main__':
    startup_json = comm.receive()
    startup_msg = comm.json_decode(startup_json)

    if not startup_msg.get('manual', None) == None:
        solver_name = startup_msg['manual']
        manual_request(solver_name)

    elif not startup_msg.get('generate_folds', None) == None:
        import optunity.cross_validation as cv
        cv_opts = startup_msg['generate_folds']
        fold_request(cv_opts)

    elif startup_msg.get('maximize', None) or startup_msg.get('minimize', None):
        if startup_msg.get('maximize', False):
            kwargs = startup_msg['maximize']
            solve_fun = optunity.maximize
        else:
            kwargs = startup_msg['minimize']
            solve_fun = optunity.minimize

        max_or_min(solve_fun, kwargs,
                   startup_msg.get('constraints', {}),
                   startup_msg.get('default', None),
                   startup_msg.get('call_log', None))


    else:  # solving a given problem
        mgr = comm.EvalManager()
        func = optunity.wrap_constraints(comm.make_piped_function(mgr),
                                         startup_msg.get('default', None),
                                         **startup_msg.get('constraints', {})
                                         )

        if startup_msg.get('call_log', False):
            func = optunity.wrap_call_log(func, startup_msg['call_log'])
        else:
            func = functions.logged(func)

        maximize = startup_msg.get('maximize', True)

        # instantiate solver
        try:
            solver = optunity.make_solver(startup_msg['solver'],
                                          **startup_msg['config'])
        except KeyError:
            msg = {'error_msg': 'Unable to instantiate solver.'}
            comm.send(comm.json_encode(msg))
            print(startup_msg, file=sys.stderr)
            exit(1)
        except EOFError:
            msg = {'error_msg': 'Broken pipe.'}
            comm.send(comm.json_encode(msg))
            exit(1)

        # solve and send result
        try:
            solution, rslt = optunity.optimize(solver, func, maximize,
                                               pmap=mgr.pmap)
        except EOFError:
            msg = {'error_msg': 'Broken pipe.'}
            comm.send(comm.json_encode(msg))
            exit(1)

        result = rslt._asdict()
        result['solution'] = solution
        result_json = comm.json_encode(result)
        comm.send(result_json)
        exit(0)
