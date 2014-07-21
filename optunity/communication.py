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


def receive(channel=sys.stdin):
    """Reads data from channel."""
    line = channel.readline()[:-1]
    if not line:
        raise EOFError("Unexpected end of communication.")
    return line


class EvalManager(object):

    def __init__(self):
        self._vectorized = False
        self._queue = []
        self._record = []

    def pmap(self, f, *args):
        self._queue = []
        self._vectorized = True
        # fill the queue

        try:
            non_piped = [f(*a) for a in zip(*args)]
        except functions.MaximumEvaluationsException:
            if not len(self.queue):
                raise functions.MaximumEvaluationsException()

        # complete list of results: some evaluations may not have reached
        # the manager: namely violated constraints or calls that were
        # already in a call log
        results = self.flush_queue()
        if results is None:
            results = non_piped
        else:
            results = [y if x == None or x == (None,) else x
                       for x, y in zip(non_piped, itertools.cycle(results))]
        self._record.extend(results)

        self._vectorized = False
        self._queue = []
        return results

    def pipe_eval(self, **kwargs):
        json_data = json_encode(kwargs)
        send(json_data)

        json_reply = receive()
        decoded = json_decode(json_reply)
        if decoded.get('error', False):  # TODO: allow error handling higher up?
            sys.stderr('ERROR: ' + decoded['error'])
            sys.exit(1)
        self._record.append(decoded['value'])
        return decoded['value']

    def add_to_queue(self, **kwargs):
        self._queue.append(kwargs)

    @property
    def vectorized(self):
        return self._vectorized

    @property
    def queue(self):
        return self._queue

    @property
    def record(self):
        return self._record

    def flush_queue(self):
        if self.queue:
            json_data = json_encode(self.queue)
            send(json_data)

            json_reply = receive()
            decoded = json_decode(json_reply)
            if decoded.get('error', False):
                sys.stderr('ERROR: ' + decoded['error'])
                sys.exit(1)
            return decoded['values']
        return None


def make_piped_function(mgr):
    def piped_function_eval(**kwargs):  # TODO: *args ?
        """Returns a function evaluated through a pipe with arguments args.

        args must be a namedtuple."""
        if mgr.vectorized:
            mgr.add_to_queue(**kwargs)
            return None
        else:
            return mgr.pipe_eval(**kwargs)
    return piped_function_eval


if __name__ == '__main__':
    import doctest
    doctest.testmod()
