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

This standalone subprocess can communicate through stdin/stdout or sockets. To use sockets:

- **standalone as client**: specify the port as first commandline argument and host as second (omitting host will imply `localhost`).

    .. code::

        python -m optunity.standalone <PORT> <HOST>


- **standalone as server**: launch with 'server' as first command line argument. The port number that is being listened on will be printed on stdout.

    .. code::

        python -m optunity.standalone server


Requesting manuals
-------------------

Emulates :func:`optunity.manual`.

Request:

+--------+------------------------------------------------------------+----------+
| Key    | Value                                                      | Optional |
+========+============================================================+==========+
| manual | name of the solver whose manual we want or empty string    | no       |
+--------+------------------------------------------------------------+----------+

Examples::

    {"manual": ""}
    {"manual": "grid search"}

Reply:

+--------------+-----------------------------------------------+-----------------+
| Key          | Value                                         | Type            |
+==============+===============================================+=================+
| manual       | the manual that was requested                 | list of strings |
+--------------+-----------------------------------------------+-----------------+
| solver_names | the name of the solver whose manual was       | list of strings |
|              | requested or a list of all registered solvers |                 |
+--------------+-----------------------------------------------+-----------------+

Manuals are returned as a list of strings, each string is a line of the manual.

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

Example::

    {"generate_folds": {"num_instances": 10, "num_folds": 2, "num_iter": 5, "strata": [[1, 2], [3, 4]], "clusters": [[5, 6], [7, 8]]}}

Reply:

+--------------+---------------------+---------------------------------------------------------+
| Key          | Value               | Type                                                    |
+==============+=====================+=========================================================+
| folds        | the resulting folds | list (iterations) of lists (folds) of lists (instances) |
+--------------+---------------------+---------------------------------------------------------+

The inner lists contain the instance indices per fold (0-based indexing).

Example::

    {"folds": [[[2, 3, 0, 5, 8], [1, 4, 7, 6, 9]], [[2, 4, 7, 8, 0], [1, 3, 6, 9, 5]], [[2, 4, 9, 7, 8], [1, 3, 0, 5, 6]], [[1, 3, 7, 6, 5], [2, 4, 9, 8, 0]], [[2, 3, 5, 8, 0], [1, 4, 6, 9, 7]]]}


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

+-------------+-------------------------------------------------------------+----------+
| Key         | Value                                                       | Optional |
+=============+=============================================================+==========+
| maximize    | dictionary:                                                 | no       |
|             |                                                             |          |
|             | - **num_evals** number of permitted function evaluations    | - no     |
|             | - **box constraints** dictionary                            | - no     |
+-------------+-------------------------------------------------------------+----------+
| call_log    | a call log of previous function evaluations                 | yes      |
+-------------+-------------------------------------------------------------+----------+
| constraints | domain constraints on the objective function                | yes      |
|             |                                                             |          |
|             | - **ub_{oc}** upper bound (open/closed)                     | - yes    |
|             | - **lb_{oc}** lower bound (open/closed)                     | - yes    |
|             | - **range_{oc}{oc}** interval bounds                        | - yes    |
+-------------+-------------------------------------------------------------+----------+
| default     | number, default function value if constraints are violated  | yes      |
+-------------+-------------------------------------------------------------+----------+


After the initial setup message, Optunity will send objective function evaluation requests.
These request may be for a scalar or a vector evaluation, and look like this:

**Scalar evaluation request**: the message is a dictionary containing the hyperparameter
names as keys and their associated values as values.

An example request to evaluate f(x, y) in (x=1, y=2)::

    {"x": 1, "y": 2}


**Vector evaluation request**: the message is a list of dictionaries with the same form as
the dictionary of a scalar evaluation request.

An example request to evaluate f(x, y) in (x=1, y=2) and (x=2, y=3)::

    [{"x": 1, "y": 2}, {"x": 2, "y": 3}]

The replies to evaluation requests are simple:

- scalar request: dictionary with key *value* and value the objective function value
- vector request: dictionary with key *values* and value a list of function values

.. note::

    Results of vector evaluations must be returned in the same order as the request.


When a solution is found, Optunity will send a final message with all necessary information
and then exit. This final message contains the following:

+----------+--------------------------------------------------------+--------------+
| Key      | Value                                                  | Type         |
+==========+========================================================+==============+
| solution | the optimal hyperparameters                            | dictionary   |
+----------+--------------------------------------------------------+--------------+
| details  | various details about the solving process              | dictionary   |
|          |                                                        |              |
|          | - **optimum**: f(solution)                             | - number     |
|          | - **stats**: number of evaluations and wall clock time | - dictionary |
|          | - **call_log**: record of all function evaluations     | - dictionary |
|          | - **report**: optional solver report                   | - optional   |
+----------+--------------------------------------------------------+--------------+
| solver   | information about the solver that was used             | dictionary   |
+----------+--------------------------------------------------------+--------------+

Minimize
---------

Identical to maximize (above) except that the initial message has the key
``minimize`` instead of ``maximize``.

Emulates :func:`optunity.minimize`.


Make solver
------------

Attempt to instantiate a solver from given configuration. This serves as a sanity-check.

Emulates :func:`optunity.make_solver`.

+-------------+---------------------------------------------------------+----------+
| Key         | Value                                                   | Optional |
+=============+=========================================================+==========+
| make_solver | dictionary                                              | no       |
|             |                                                         |          |
|             | - **solver_name** name of the solver to be instantiated | - no     |
|             | - everything necessary for the solver constructor       | - no     |
|             |                                                         |          |
|             | see :doc:`/api/optunity.solvers` for details.           |          |
+-------------+---------------------------------------------------------+----------+

Example::

    {"make_solver": {"x": [1, 2], "y": [2, 3], "solver_name": "grid search"}}


Optunity replies with one of two things:

- ``{"success": true}``: the solver was correctly instantiated
- ``{"error_msg": "..."}``: instantiating the solver failed

Optimize
---------

Using the optimize functionality involves sending an initial message with the
arguments to :func:`optunity.optimize` as shown below.

Subsequently, the solver will send objective function evaluation requests sequentially.
These requests can be for scalar or vector evaluations (details below).

When the solution is found, or the
maximum number of evaluations is reached a final message is sent with all details.

Emulates :func:`optunity.optimize`.

+-------------+-----------------------------------------------------------------+----------+
| Key         | Value                                                           | Optional |
+=============+=================================================================+==========+
| optimize    | dictionary                                                      | no       |
|             |                                                                 |          |
|             | - **max_evals**: maximum number of evaluations, or 0            | - yes    |
|             | - **maximize**: boolean, indicates maximization (default: true) | - yes    |
+-------------+-----------------------------------------------------------------+----------+
| solver      | dictionary                                                      | no       |
|             |                                                                 |          |
|             | - **solver_name**: name of the solver                           | - no     |
|             | - everything necessary for the solver constructor               | - no     |
|             |                                                                 |          |
|             | see :doc:`/api/optunity.solvers` for details.                   |          |
+-------------+-----------------------------------------------------------------+----------+
| call_log    | a call log of previous function evaluations                     | yes      |
+-------------+-----------------------------------------------------------------+----------+
| constraints | domain constraints on the objective function                    | yes      |
|             |                                                                 |          |
|             | - **ub_{oc}** upper bound (open/closed)                         | - yes    |
|             | - **lb_{oc}** lower bound (open/closed)                         | - yes    |
|             | - **range_{oc}{oc}** interval bounds                            | - yes    |
+-------------+-----------------------------------------------------------------+----------+
| default     | number, default function value if constraints are violated      | yes      |
+-------------+-----------------------------------------------------------------+----------+

After the initial setup message, Optunity will send objective function evaluation requests.
These request may be for a scalar or a vector evaluation, and look like this:

**Scalar evaluation request**: the message is a dictionary containing the hyperparameter
names as keys and their associated values as values.

An example request to evaluate f(x, y) in (x=1, y=2)::

    {"x": 1, "y": 2}


**Vector evaluation request**: the message is a list of dictionaries with the same form as
the dictionary of a scalar evaluation request.

An example request to evaluate f(x, y) in (x=1, y=2) and (x=2, y=3)::

    [{"x": 1, "y": 2}, {"x": 2, "y": 3}]

The replies to evaluation requests are simple:

- scalar request: dictionary with key *value* and value the objective function value
- vector request: dictionary with key *values* and value a list of function values

.. note::

    Results of vector evaluations must be returned in the same order as the request.


When a solution is found, Optunity will send a final message with all necessary information
and then exit. This final message contains the following:

+----------+--------------------------------------------------------+--------------+
| Key      | Value                                                  | Type         |
+==========+========================================================+==============+
| solution | the optimal hyperparameters                            | dictionary   |
+----------+--------------------------------------------------------+--------------+
| details  | various details about the solving process              | dictionary   |
|          |                                                        |              |
|          | - **optimum**: f(solution)                             | - number     |
|          | - **stats**: number of evaluations and wall clock time | - dictionary |
|          | - **call_log**: record of all function evaluations     | - dictionary |
|          | - **report**: optional solver report                   | - optional   |
+----------+--------------------------------------------------------+--------------+



.. moduleauthor:: Marc Claesen

"""

from __future__ import print_function
import sys
import keyword

# optunity imports
from . import communication as comm
from . import functions
import optunity

# list of all Python keywords, which may be used as argument names in wrappers
_illegal_keys = keyword.kwlist


def _change_keys_in_solver_config(cfg):
    replacements = comm._find_replacements(_illegal_keys, solver_config)
    return comm._replace_keys(solver_config, replacements), replacements


def manual_request(solver_name):
    """Emulates :func:`optunity.manual`."""
    if len(solver_name) == 0:
        solver_name = None
    try:
        manual, solver_names = optunity.api._manual_lines(solver_name)
    except (ValueError, KeyError):
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
    except (KeyError, ValueError):
        msg = {'error_msg': 'number of instances num_instances must be set.'}
        comm.send(comm.json_encode(msg))
        print(startup_msg, file=sys.stderr)
        exit(1)

    num_folds = cv_opts.get('num_folds', 10)
    strata = cv_opts.get('strata', None)
    clusters = cv_opts.get('clusters', None)
    num_iter = cv_opts.get('num_iter', 1)

    folds = [optunity.generate_folds(num_instances, num_folds=num_folds,
                                     strata=strata, clusters=clusters)
             for _ in range(num_iter)]

    msg = {'folds': folds}
    comm.send(comm.json_encode(msg))
    exit(0)


def make_solver(solver_config):
    try:
        optunity.make_solver(**solver_config)
    except (KeyError, ValueError, TypeError) as e:
        msg = {'error_msg': 'Unable to instantiate solver: ' + str(e)}
        comm.send(comm.json_encode(msg))
        print(solver_config, file=sys.stderr)
        exit(1)

    msg = {'success': 'true'}
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
    replacements = comm._find_replacements(_illegal_keys, kwargs)
    solver_config = comm._replace_keys(kwargs, replacements)
    constraints = comm._replace_keys(constraints, replacements)

    # prepare objective function
    mgr = comm.EvalManager(replacements=replacements)
    func = prepare_fun(mgr, constraints, default, call_log)

    # solve problem
    try:
        solution, rslt, solver = solve_fun(func, pmap=mgr.pmap, **solver_config)
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


def optimize(solver_config, constraints, default, call_log, maximize, max_evals):
    """Emulates :func:`optunity.optimize`."""
    replacements = comm._find_replacements(_illegal_keys, solver_config)
    solver_config = comm._replace_keys(solver_config, replacements)
    constraints = comm._replace_keys(constraints, replacements)

    # prepare objective function
    mgr = comm.EvalManager(replacements=replacements)
    func = prepare_fun(mgr, constraints, default, call_log)

    # make the solver
    try:
        solver = optunity.make_solver(**solver_config)
    except (ValueError, KeyError) as e:
        msg = {'error_msg': 'Unable to instantiate solver: ' + str(e)}
        comm.send(comm.json_encode(msg))
        print(solver_config, file=sys.stderr)
        exit(1)

    # solve problem
    try:
        solution, rslt = optunity.optimize(solver, func, maximize=maximize,
                                           max_evals=max_evals, pmap=mgr.pmap)
    except EOFError:
        msg = {'error_msg': 'Broken pipe.'}
        comm.send(comm.json_encode(msg))
        exit(1)

    # send solution and exit
    result = rslt._asdict()
    result['solution'] = solution
    result_json = comm.json_encode(result)
    comm.send(result_json)
    exit(0)


def main():

    # open a socket if port [+ host] specified in commandline args
    if len(sys.argv) > 1:
        if sys.argv[1] == 'server':
            port, server_socket = comm.open_server_socket()
            print(port)
            ## flush is needed for R pipe():
            sys.stdout.flush()
            comm.accept_server_connection(server_socket)

        else:
            try:
                port = int(sys.argv[1])
            except ValueError as e:
                print('Invalid socket port: ' + str(e))
                sys.exit(1)
            if len(sys.argv) > 2:
                host = sys.argv[2]
            else:
                host = 'localhost'
            comm.open_socket(port, host)

    startup_json = comm.receive()

    try:
        startup_msg = comm.json_decode(startup_json)

        if 'manual' in startup_msg:
            solver_name = startup_msg['manual']
            manual_request(solver_name)

        elif 'generate_folds' in startup_msg:
            import optunity.cross_validation as cv
            cv_opts = startup_msg['generate_folds']
            fold_request(cv_opts)

        elif 'make_solver' in startup_msg:
            solver_config = startup_msg['make_solver']
            replacements = comm._find_replacements(_illegal_keys, solver_config)
            solver_config = comm._replace_keys(solver_config, replacements)
            make_solver(solver_config)

        elif 'maximize' in startup_msg or 'minimize' in startup_msg:
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

        elif 'optimize' in startup_msg:
            max_evals = startup_msg['optimize'].get('max_evals', 0)
            maximize = startup_msg['optimize'].get('maximize', True)

            # sanity check
            if not 'solver' in startup_msg:
                msg = {'error_msg': 'No solver specified in startup message.'}
                comm.send(comm.json_encode(msg))
                print(startup_msg, file=sys.stderr)
                exit(1)

            optimize(startup_msg['solver'],
                    startup_msg.get('constraints', {}),
                    startup_msg.get('default', None),
                    startup_msg.get('call_log', None),
                    maximize, max_evals)

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
            except (ValueError, KeyError):
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

    except (ValueError, TypeError, AttributeError) as e:
        msg = {'error_msg': str(e)}
        comm.send(comm.json_encode(msg))
        exit(1)


if __name__ == '__main__':
    main()
