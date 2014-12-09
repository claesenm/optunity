function optunity_comm_close_subprocess( m2py, py2m, pid )
%CLOSE_SUBPROCESS Closes all communication and waits for subprocess to die.

fclose(m2py);
fclose(py2m);
[pid, status, msg] = waitpid(pid);

end
