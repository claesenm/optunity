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


from __future__ import print_function
import sys

# optunity imports
from optunity import communication as comm
import optunity


startup_json = comm.receive()
startup_msg = comm.json_decode(startup_json)

if startup_msg.get('manual', False):
    solver_name = startup_msg.get('solver', None)
    try:
        manual, solver_names = optunity.manual_request(solver_name)
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

elif startup_msg.get('cross_validation', False):
    cv_opts = startup_msg['cross_validation']

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
    x = [None] * num_instances

    @optunity.cross_validated(x, num_folds, strata=strata, clusters=clusters,
                              num_iter=num_iter)
    def f():
        pass

    msg = {'folds': f.folds}
    comm.send(comm.json_encode(msg))
    exit(0)

else:  # solving a given problem
    func = optunity.wrap_constraints(comm.piped_function_eval,
                                startup_msg.get('constraints', None),
                                startup_msg.get('default', None))

    if startup_msg.get('call_log', False):
        func = optunity.wrap_call_log(func, startup_msg['call_log'])

    # instantiate solver
    try:
        solver = optunity.make_solver(startup_msg['solver'],
                                 startup_msg['config'])
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
        solution, optimum, num_evals, call_log, report = optunity.maximize(solver, func)
    except EOFError:
        msg = {'error_msg': 'Broken pipe.'}
        comm.send(comm.json_encode(msg))
        exit(1)

    result = {'solution': solution, 'optimum': optimum,
              'num_evals': num_evals}

    if report:
        result['report'] = report

    if startup_msg.get('return_call_log', False):
        result['call_log'] = call_log

    result_json = comm.json_encode(result)
    comm.send(result_json)
    exit(0)
