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
import itertools
from . import functions
from . import parallel

import multiprocessing
import threading

__DEBUG = False


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


def send(data, channel=sys.stdout):
    """Writes data to channel and flushes."""
    print(data, file=channel)
    channel.flush()
    return None


def receive(channel=sys.stdin):
    """Reads data from channel."""
    line = channel.readline()[:-1]
    if not line:
        raise EOFError("Unexpected end of communication.")
    return line


class EvalManager(object):

    def __init__(self):
        # are we doing a parallel function evaluation?
        self._vectorized = False

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
        json_data = json_encode(kwargs)
        send(json_data)

        json_reply = receive()
        decoded = json_decode(json_reply)
        if decoded.get('error', False):  # TODO: allow error handling higher up?
            sys.stderr('ERROR: ' + decoded['error'])
            sys.exit(1)
        return decoded['value']

    def add_to_queue(self, **kwargs):
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
