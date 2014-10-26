function [ solver ] = make_solver( name, varargin )
%MAKE_SOLVER Attempts to construct a solver with given name and options.
% This function accepts the following arguments:
% - name: name of the solver
% - varargin: should be a list of key:value arguments as explained in each
%   solver's manual. 
%
% Please refer to optunity.manual() for further details on solver options.
% An error is generated if construction of the solver fails.

%% process varargin
assert(mod(numel(varargin), 2) == 0, ... 
    'varargin should be of the form [key,value,...]');

cfg = struct(varargin{:});
cfg.solver_name = name;
msg = struct('make_solver', cfg);

%% launch SOAP subprocess
[m2py, py2m, stderr, subprocess, cleaner] = optunity.comm.launch();

%% attempt to initialize solver
json_request = optunity.comm.json_encode(msg);
optunity.comm.writepipe(m2py, json_request);

json_reply = optunity.comm.readpipe(py2m);
reply = optunity.comm.json_decode(json_reply);

if isfield(reply, 'error_msg')
    error(['Error: ', reply.error_msg]);
end

solver = optunity.Solver(name, cfg);

end