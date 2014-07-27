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

Request:

+-------------+----------------------------------------------------------+----------+
| Parameter   | Description                                              | Optional |
+=============+==========================================================+==========+
| solver_name | (string) name of the solver whose manual is requested    | yes      |
+-------------+----------------------------------------------------------+----------+

Reply:

+--------------+------------------------------------------+-----------------+
| Parameter    | Description                              | Type            |
+==============+==========================================+=================+
| manual       | the manual that was requested            | list of strings |
| solver_names | the names of all registered solver,      | list of strings |
|              | or the solver used in the manual request |                 |
+--------------+------------------------------------------+-----------------+


Maximize or minimize
-----------------------

Optimize
---------


Generate cross-validation folds
---------------------------------


.. moduleauthor:: Marc Claesen

"""

from __future__ import print_function
import sys

# optunity imports
from . import communication as comm
from . import functions
import optunity


if __name__ == '__main__':
    startup_json = comm.receive()
    startup_msg = comm.json_decode(startup_json)

    if startup_msg.get('manual', False):
        solver_name = startup_msg.get('solver', None)
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

    elif startup_msg.get('generate_folds', False):
        import optunity.cross_validation as cv
        cv_opts = startup_msg['generate_folds']

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

    elif startup_msg.get('maximize', None) or startup_msg.get('minimize', None):
        if startup_msg.get('maximize', False):
            kwargs = startup_msg['maximize']
            solve_fun = optunity.maximize
        else:
            kwargs = startup_msg['minimize']
            solve_fun = optunity.minimize

        # prepare objective function
        mgr = comm.EvalManager()
        func = optunity.wrap_constraints(comm.make_piped_function(mgr),
                                         startup_msg.get('default', None),
                                         **startup_msg.get('constraints', {})
                                         )
        if startup_msg.get('call_log', False):
            func = optunity.wrap_call_log(func, startup_msg['call_log'])
        else:
            func = functions.logged(func)

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
