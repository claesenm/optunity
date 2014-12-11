function [ sock, pid, cleaner ] = comm_launch()
%LAUNCH Wrapper around all logic involving the launching of Optunity.
%   Optunity is launched through a Java Runtime().exec() call.
%   To enable Optunity to locate installed libraries and necessary
%   dependencies, we must pass Python's paths explicitly to the subprocess
%   environment.
%
%   This function returns stdin, stdout, stderr and a handler of the
%   Optunity subprocess. 
%   Destruction of the subprocess must be done manually.


% hack to fix empty paths when spawning Optunity
% find the current pathused by python
% necessary to pass as an env variable when launching Optunity
persistent env
% persistent path
if isempty(env)
    pkg load sockets
    path = mfilename('fullpath');
    f = filesep;
    hit = [f, 'wrappers', f, 'octave', f, 'optunity', f, 'comm', f, 'comm_launch'];
    path= path(1:strfind(path, hit));
    path_to_bin = [path, f, 'bin'];

    [~, pathstr] = system(['python ', path_to_bin, f, 'print_system_path.py']);
    pathstr = pathstr(pathstr ~= '''');
    pathstr = pathstr(pathstr ~= '[');
    pathstr = pathstr(pathstr ~= ']');
    pathstr = strrep(pathstr, ', ', pathsep);
    env = strtrim(pathstr);

    % attach s path to env
    env = [strtrim(env), pathsep, strtrim(path)];
    setenv("PYTHONPATH", env);
end

% open server socket
servSock = socket();
port = 0;
do
    port = randi([1025, 50000]);
    r = bind(servSock, port);
until r >= 0
r = listen(servSock, 0);
cmd = ['python -m optunity.standalone ', num2str(port)];

pid = popen(cmd, "r");
fflush(stdout);
[sock, info] = accept(servSock);
disconnect(servSock);
close(servSock);

if pid == -1
    error("Error launching Optunity back-end.");
end

% provide RAII-style automatic cleanup when cleaner goes out of scope
% e.g. both upon normal caller exit or an error
cleaner = onCleanup(@()comm_close_subprocess(sock, pid));
end
