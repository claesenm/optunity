from __future__ import print_function
import subprocess
import socket
import sys


# prepare server socket
print('Making server socket.')
serv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serv_sock.bind(('', 0))
serv_sock.listen(0)
port = serv_sock.getsockname()[1]

# launch piped
print('Launching piped subprocess.')
piped = subprocess.Popen(['python', '-m', 'optunity.piped', str(port)])

print('Connected to subprocess.')
sock = serv_sock.accept()[0]
__channel_in = sock.makefile('r')
__channel_out = sock.makefile('w')

# check if we want to run a custom command
if len(sys.argv) > 1:
    msg = sys.argv[1]
else:
    msg = '{"manual": ""}'

print('Sending message: ' + msg)
print(msg, file=__channel_out)
__channel_out.flush()

reply = __channel_in.readline()
print('Received repy: ' + reply)
