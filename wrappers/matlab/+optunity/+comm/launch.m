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
if isempty(env)
    [~, pathstr] = system('python -c "import sys; print(sys.path)"');
    pathstr = pathstr(pathstr ~= '''');
    pathstr = pathstr(pathstr ~= '[');
    pathstr = pathstr(pathstr ~= ']');
    pathstr = strrep(pathstr, ', ',':');
    env = ['PYTHONPATH=', pathstr];
    
    % attach optunity's path to env
    path = mfilename('fullpath');
    path = path(1:strfind(path, '/wrappers/matlab/launch_optunity'));
    env = [env, ':', path];
end

cmd = 'python -m optunity.piped';
[m2py, py2m, stderr, handle] = optunity.comm.popen( cmd, env );
cleaner = onCleanup(@()optunity.comm.close_subprocess(m2py, py2m, stderr, ...
    handle));
end