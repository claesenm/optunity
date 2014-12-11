function comm_close_subprocess( sock, pid )
%CLOSE_SUBPROCESS Closes all communication and waits for subprocess to die.

disconnect(sock);
close(sock);
fclose(pid);
%[err, msg] = kill (pid, 15);

end
