function [ m2py, py2m, stderr, pid, cleaner ] = optunity_comm_launch()
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

% http://octave.1599824.n4.nabble.com/example-how-to-get-popen2-working-XP-MSVC-2-9-15-octave-td1628494.html
[m2py, py2m, pid] = popen2("python", {"-m", "optunity.standalone"}, true);

% provide RAII-style automatic cleanup when cleaner goes out of scope
% e.g. both upon normal caller exit or an error
cleaner = onCleanup(@()optunity_comm_close_subprocess(m2py, py2m, pid));
stderr = '';
end
