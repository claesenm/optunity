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
import json
import sys
import socket
import itertools
from . import functions
from . import parallel
import math


import multiprocessing
import threading

__DEBUG = False


__channel_in = sys.stdin
__channel_out = sys.stdout


def _find_replacement(key, kwargs):
    """Finds a replacement for key that doesn't collide with anything in kwargs."""
    key += '_'
    while key in kwargs:
        key += '_'
    return key


def _find_replacements(illegal_keys, kwargs):
    """Finds replacements for all illegal keys listed in kwargs."""
    replacements = {}
    for key in illegal_keys:
        if key in kwargs:
            replacements[key] = _find_replacement(key, kwargs)
    return replacements


def _replace_keys(kwargs, replacements):
    """Replace illegal keys in kwargs with another value."""

    for key, replacement in replacements.items():
        if key in kwargs:
            kwargs[replacement] = kwargs[key]
            del kwargs[key]
    return kwargs


def json_encode(data):
    """Encodes given data in a JSON string."""
    return json.dumps(data)


def json_decode(data):
    """Decodes given JSON string and returns its data.

    >>> orig = {1: 2, 'a': [1,2]}
    >>> data = json_decode(json_encode(orig))
    >>> data[str(1)]
    2
    >>> data['a']
    [1, 2]
    """
    return json.loads(data)


def send(data):
    """Writes data to channel and flushes."""
    print(data, file=__channel_out)
    __channel_out.flush()


def receive():
    """Reads data from channel."""
    line = __channel_in.readline()[:-1]
    if not line:
        raise EOFError("Unexpected end of communication.")
    return line


def open_socket(port, host='localhost'):
    """Opens a socket to host:port and reconfigures internal channels."""
    global __channel_in
    global __channel_out
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        sock.connect((host, port))
        __channel_in = sock.makefile('r')
        __channel_out = sock.makefile('w')
    except (socket.error, OverflowError, ValueError) as e:
        print('Error making socket: ' + str(e), file=sys.stderr)
        sys.exit(1)


def open_server_socket():
    serv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        serv_sock.bind(('', 0))
    except socket.err as msg:
        print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
        sys.exit()
    serv_sock.listen(0)
    port = serv_sock.getsockname()[1]
    return port, serv_sock


def accept_server_connection(server_socket):
    global __channel_in
    global __channel_out
    try:
        sock, _ = server_socket.accept()
        __channel_in = sock.makefile('r')
        __channel_out = sock.makefile('w')
    except (socket.error, OverflowError, ValueError) as e:
        print('Error making socket: ' + str(e), file=sys.stderr)
        sys.exit(1)


class EvalManager(object):

    def __init__(self, max_vectorized=100, replacements={}):
        """Constructs an EvalManager object.

        :param max_vectorized: the maximum size of a vector evaluation
            larger vectorizations will be chunked
        :type max_vectorized: int
        :param replacements: a mapping of `original:replacement` keyword names
        :type replacements: dict

        """
        # are we doing a parallel function evaluation?
        self._vectorized = False
        self._max_vectorized = max_vectorized

        # the queue used for parallel evaluations
        self._queue = None

        # lock to use to add evaluations to the queue
        self._queue_lock = multiprocessing.Lock()

        # lock to get results
        self._result_lock = multiprocessing.Lock()

        # used to check whether Future started running
        self._semaphore = None

        # used to signal Future's to get their result
        self._processed_semaphore = None

        # keys that must be replaced
        self._replacements = dict((v, k) for k, v in replacements.items())


    @property
    def replacements(self):
        """Keys that must be replaced: keys are current values, values are
        what must be sent through the pipe."""
        return self._replacements

    @property
    def max_vectorized(self):
        return self._max_vectorized

    @property
    def cv(self):
        return self._cv

    @property
    def semaphore(self):
        return self._semaphore

    @property
    def processed_semaphore(self):
        return self._processed_semaphore

    def pmap(self, f, *args):
        """Performs vector evaluations through pipes.

        :param f: the objective function (piped_function_eval)
        :type f: callable
        :param args: function arguments
        :type args: iterables

        The vector evaluation is sent in chunks of size self.max_vectorized.

        """
        argslist = zip(*args)
        results = []

        # partition the vector evaluation in chunks <= self.max_vectorized
        for idx in range(int(math.ceil(1.0 * len(argslist) / self.max_vectorized))):
            chunk = argslist[idx * self.max_vectorized:(idx + 1) * self.max_vectorized]
            results.extend(self._vector_eval(f, *zip(*chunk)))

        return results

    def _vector_eval(self, f, *args):
        self._queue = []
        self._vectorized = True
        self._semaphore = multiprocessing.Semaphore(0)
        self._processed_semaphore = multiprocessing.Semaphore(0)

        # make sure correct evaluations lead to counter increments
        def wrap(*args):
            result = f(*args)
            # signal mgr to add next future
            self.semaphore.release()
            return result

        # create list of futures
        futures = []
        for ar in zip(*args):
            futures.append(parallel.Future(wrap, *ar))
            try:
                self.semaphore.acquire()
            except functions.MaximumEvaluationsException:
                if not len(self.queue):
                    # FIXME: some threads may be running atm
                    raise
                break

        # process queue
        self.flush_queue()

        # notify all waiting futures
        for _ in range(len(self.queue)):
            self.processed_semaphore.release()

        # gather results
        results = [f() for f in futures]
        for f in futures:
            f.join()
        return results

    def pipe_eval(self, **kwargs):
        # fix python keywords back to original
        kwargs = _replace_keys(kwargs, self.replacements)

        json_data = json_encode(kwargs)
        send(json_data)

        json_reply = receive()
        decoded = json_decode(json_reply)
        if decoded.get('error', False):  # TODO: allow error handling higher up?
            sys.stderr('ERROR: ' + decoded['error'])
            sys.exit(1)
        return decoded['value']

    def add_to_queue(self, **kwargs):
        kwargs = _replace_keys(kwargs, self.replacements)
        try:
            self.queue_lock.acquire()
            self.queue.append(kwargs)
            idx = len(self.queue) - 1
        finally:
            self.queue_lock.release()
        return idx

    @property
    def vectorized(self):
        return self._vectorized

    @property
    def queue(self):
        return self._queue

    @property
    def queue_lock(self):
        return self._queue_lock

    def get(self, number):
        try:
            self._result_lock.acquire()
            value = self._results[number]
        finally:
            self._result_lock.release()
        return value

    def flush_queue(self):
        self._results = []
        if self._queue:
            json_data = json_encode(self.queue)
            send(json_data)

            json_reply = receive()
            decoded = json_decode(json_reply)
            if decoded.get('error', False):
                sys.stderr('ERROR: ' + decoded['error'])
                sys.exit(1)
            self._results = decoded['values']


def make_piped_function(mgr):
    def piped_function_eval(**kwargs):
        """Returns a function evaluated through a pipe with arguments args.

        args must be a namedtuple."""
        if mgr.vectorized:
            # add to mgr queue
            number = mgr.add_to_queue(**kwargs)
            # signal mgr to continue
            mgr.semaphore.release()

            # wait for results
            mgr.processed_semaphore.acquire()
            return mgr.get(number)
        else:
            return mgr.pipe_eval(**kwargs)
    return piped_function_eval


if __name__ == '__main__':
    import doctest
    doctest.testmod()
