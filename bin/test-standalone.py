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

# This executable is meant to debug optunity.piped across platforms.
# It launches optunity as any alien environment would and communicates
# over sockets.

from __future__ import print_function
import subprocess
import socket
import sys

# prepare server socket
print('Making server socket.')
serv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    serv_sock.bind(('', 0))
except socket.err as msg:
    print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
    sys.exit()

serv_sock.listen(0)
port = serv_sock.getsockname()[1]

# launch piped
print('Launching piped subprocess (port ' + str(port) + ').')
piped = subprocess.Popen(['python', '-m', 'optunity.standalone', str(port)])

p = piped.poll()
print('Poll result: ' + str(p))

print('Connected to subprocess.')
sock, address = serv_sock.accept()

p = piped.poll()
print('Poll result: ' + str(p))

print('Received connection from ' + str(address))
__channel_in = sock.makefile('r')
__channel_out = sock.makefile('w')


if len(sys.argv) > 1:
    msg = sys.argv[1]
else:
    msg = '{"manual": ""}'

print('Sending message: ' + msg)
print(msg, file=__channel_out)
__channel_out.flush()

reply = __channel_in.readline()
print('Received repy: ' + reply)
