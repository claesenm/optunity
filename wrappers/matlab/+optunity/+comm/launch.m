function [ m2py, py2m, stderr, handle, cleaner ] = launch()
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
% find the current path used by python
% necessary to pass as an env variable when launching Optunity
persistent env
persistent path_to_bin
if isempty(env)
    path = mfilename('fullpath');
    path = path(1:strfind(path, '/wrappers/matlab/+optunity/+comm/launch'));
    path_to_bin = [path,'/bin/'];
    
    [~, pathstr] = system([path_to_bin, 'print_system_path.py']);
    pathstr = pathstr(pathstr ~= '''');
    pathstr = pathstr(pathstr ~= '[');
    pathstr = pathstr(pathstr ~= ']');
    pathstr = strrep(pathstr, ', ',':');
    env = ['PYTHONPATH=', pathstr];
    
    % attach optunity's path to env
    env = [env, ':', path];
end

% cmd = [path_to_bin, 'run_optunity_piped.py'];
cmd = 'python -m optunity.piped';
[m2py, py2m, handle, socket] = optunity.comm.popen( cmd, env );

% provide RAII-style automatic cleanup when cleaner goes out of scope
% e.g. both upon normal caller exit or an error
cleaner = onCleanup(@()optunity.comm.close_subprocess(m2py, py2m, ...
    handle, socket));

stderr = '';
end