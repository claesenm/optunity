function [ sock, pid, cleaner ] = optunity_comm_launch()
%LAUNCH Wrapper around all logic involving the launching of Optunity.
%   Optunity is launched through a Java Runtime().exec() call.
%   To enable Optunity to locate installed libraries and necessary
%   dependencies, we must pass Python's paths explicitly to the subprocess
%   optunity_environment.
%
%   This function returns stdin, stdout, stderr and a handler of the
%   Optunity subprocess. 
%   Destruction of the subprocess must be done manually.


% hack to fix empty paths when spawning Optunity
% find the current optunity_pathused by python
% necessary to pass as an optunity_env variable when launching Optunity
persistent optunity_env
% persistent optunity_path
if isempty(optunity_env)
    pkg load sockets
    optunity_path = mfilename('fullpath');
    f = filesep;
    hit = [f, 'wrappers', f, 'octave', f, '+optunity', f, '+comm', f, 'optunity_comm_launch'];
    optunity_path= optunity_path(1:strfind(optunity_path, hit));
    path_to_bin = [optunity_path, f, 'bin'];

    [~, pathstr] = system(['python ', path_to_bin, f, 'print_system_path.py']);
    pathstr = pathstr(pathstr ~= '''');
    pathstr = pathstr(pathstr ~= '[');
    pathstr = pathstr(pathstr ~= ']');
    pathstr = strrep(pathstr, ', ', pathsep);
    optunity_env = strtrim(pathstr);

    % attach optunity_s path to optunity_env
    optunity_env = [strtrim(optunity_env), pathsep, strtrim(optunity_path)];
    setenv("PYTHONPATH", optunity_env);
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
close(servSock);

if pid == -1
    error("Error launching Optunity back-end.");
end

% provide RAII-style automatic cleanup when cleaner goes out of scope
% e.g. both upon normal caller exit or an error
cleaner = onCleanup(@()optunity_comm_close_subprocess(sock, pid));
stderr = '';
end
