function [ solver ] = optunity_make_solver( name, varargin )
%MAKE_SOLVER Attempts to construct a solver with given name and options.
% This function accepts the following arguments:
% - name: name of the solver
% - varargin: should be a list of key:value arguments as explained in each
%   solver's manual. 
%
% Please refer to optunity_manual() for further details on solver options.
% An error is generated if construction of the solver fails.

%% process varargin
assert(mod(numel(varargin), 2) == 0, ... 
    'varargin should be of the form [key,value,...]');

cfg = struct(varargin{:});
cfg.solver_name = name;
msg = struct('make_solver', cfg);

%% launch SOAP subprocess
[sock, pid, cleaner] = optunity_comm_launch();

%% attempt to initialize solver
json_request = optunity_comm_json_encode(msg);
optunity_comm_writepipe(sock, json_request);

json_reply = optunity_comm_readpipe(sock);
reply = optunity_comm_json_decode(json_reply);

if isfield(reply, 'error_msg')
    error(['Error: ', reply.error_msg]);
end

solver = optunity_Solver(name, cfg);

end
