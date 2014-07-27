function [solution, details] = maximize(f, num_evals, varargin)
%MAXIMIZE: Maximizes f using the given solver and extra options.
%
% This function accepts the following arguments:
% - f: the objective function to be maximized
% - varargin: a list of optional key:value pairs
%   - solver_name: name of the solver to use (default '')
%   - parallelize: (boolean) whether or not to parallelize evaluations
%   (default true)
%   - box constraints: pairs like this: ..., 'parameter name', [lb, ub], ...

%% process varargin
defaults = struct('solver_name', '', ...
    'parallelize', true);
options = optunity.process_varargin(defaults, varargin);
parallelize = options.parallelize;
options = rmfield(options, 'parallelize');
solver_name = options.solver_name;
options = rmfield(options, 'solver_name');

%% check the box constraints
fields = fieldnames(options);
for ii=1:numel(fields)
   field = fields{ii};
   assert(numel(options.(field)) == 2 & options.(field)(1) < options.(field)(2), ...
       'invalid box constraints');
end

%% launch SOAP subprocess
[m2py, py2m, stderr, subprocess, cleaner] = optunity.comm.launch();

pipe_send = @(data) optunity.comm.writepipe(m2py, optunity.comm.json_encode(data));
pipe_receive = @() optunity.comm.json_decode(optunity.comm.readpipe(py2m));

%% initialize solver
msg = struct('solver',solver.name,'config',solver.config, ...
    'return_call_log', options.return_call_log, 'maximize', true);
if isstruct(options.constraints)
    msg.constraints = options.constraints;
    if ~isnan(options.default)
        msg.default = options.default;
    end
end
if isstruct(options.call_log)
    msg.call_log = options.call_log;
end
pipe_send(msg);

%% iteratively send function evaluation results until solved
reply = struct();
while true
    reply = pipe_receive();
    
    if isfield(reply, 'solution') || isfield(reply, 'error_msg')
        break;
    end
    
    if iscell(reply)
        results = zeros(numel(reply), 1);
        if parallelize
            parfor ii=1:numel(reply)
                results(ii) = f(reply{ii});
            end
        else
            for ii=1:numel(reply)
                results(ii) = f(reply{ii});
            end
        end
        msg = struct('values', results);
    else
        msg = struct('value', f(reply));
    end
    
    pipe_send(msg);
end

if isfield(reply, 'error_msg')
    display('Oops ... something went wrong in Optunity');
    display(['Last request: ', optunity.comm.json_encode(msg)]);
    error(reply.error_msg);
end

solution = reply.solution;
details = reply;

end