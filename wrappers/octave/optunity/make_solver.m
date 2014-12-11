function [ solver ] = make_solver( name, varargin )
%MAKE_SOLVER Attempts to construct a solver with given name and options.
% This function accepts the following arguments:
% - name: name of the solver
% - varargin: should be a list of key:value arguments as explained in each
%   solver's manual. 
%
% Please refer to manual() for further details on solver options.
% An error is generated if construction of the solver fails.

%% process varargin
assert(mod(numel(varargin), 2) == 0, ... 
    'varargin should be of the form [key,value,...]');

cfg = struct(varargin{:});
cfg.solver_name = name;
msg = struct('make_solver', cfg);

[sock, pid, cleaner] = comm_launch();

%% attempt to initialize solver
json_request = comm_json_encode(msg);
comm_writepipe(sock, json_request);

json_reply = comm_readpipe(sock);
reply = comm_json_decode(json_reply);

if isfield(reply, 'error_msg')
    error(['Error: ', reply.error_msg]);
end

solver = Solver(name, cfg);

end
