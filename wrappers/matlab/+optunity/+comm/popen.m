function [ in, out, p, socket ] = popen( cmd, env )
%POPEN Spawns a subprocess p via Java API and enables bidirectional
%communication via sockets.

%% create server socket
serverSocket = java.net.ServerSocket(0);
port = serverSocket.getLocalPort();

% append port number to optunity's launch command
cmd = [cmd, ' ', num2str(port)];

%% launch Optunity Python back-end
rt = java.lang.Runtime.getRuntime();
if numel(env) > 0
    if iscell(env)
        dim = numel(env);
        envArr = javaArray('java.lang.String',dim);
        for ii=1:dim
            envArr(ii) = java.lang.String(env{ii});
        end
    else
        envArr = javaArray('java.lang.String',1);
        envArr(1) = java.lang.String(env);
    end
    p = rt.exec(java.lang.String(cmd), envArr);
else
    p = rt.exec(java.lang.String(cmd));
end

%% create communication socket and channels
socket = serverSocket.accept();
in = java.io.PrintWriter(socket.getOutputStream(), true);
out = java.io.BufferedReader(java.io.InputStreamReader(socket.getInputStream()));

end