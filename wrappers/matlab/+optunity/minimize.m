function [solution, details, solver] = minimize(f, num_evals, varargin)
%MINIMIZE: Minimizes f in num_evals evaluations within box constraints.
%
% This function accepts the following arguments:
% - f: the objective function to be maximized
% - varargin: a list of optional key:value pairs
%   - solver_name: name of the solver to use (default '')
%   - parallelize: (boolean) whether or not to parallelize evaluations
%       (default false)
%   - box constraints: key-value pairs
%       key: hyperparameter name
%       value: [lower_bound, upper_bound]

%% process varargin
defaults = struct('solver_name', '', ...
    'parallelize', false);
options = optunity.process_varargin(defaults, varargin, false);
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
msg = options;
msg.num_evals = num_evals;
msg.solver_name = solver_name;
msg = struct('minimize', msg);
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
        
        % make sure the json becomes [] instead of a scalar
        % matlab automatically treats length-1 vectors as scalars
        if isscalar(results)
            msg = struct('values', [results, 0]); % ugly hack
        else
            msg = struct('values', results);
        end
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
solver = details.solver;
details = rmfield(details, 'solver');

end