function close_subprocess( m2py, py2m, subprocess, socket )
%CLOSE_SUBPROCESS Closes all communication and destroys subprocess.

m2py.close();
py2m.close();
subprocess.destroy();
socket.close();

end

