function [solution, details] = solve( solver_name, solver_config, f, varargin)
%SOLVE TODO: write me

%% process varargin
defaults = struct('constraints', NaN, ...
    'call_log', NaN, ...
    'return_call_log', false, ...
    'default',NaN);
options = optunity.process_varargin(defaults, varargin);

%% launch SOAP subprocess
[m2py, py2m, stderr, subprocess, cleaner] = optunity.comm.launch();

pipe_send = @(data) optunity.comm.writepipe(m2py, optunity.comm.json_encode(data));
pipe_receive = @() optunity.comm.json_decode(optunity.comm.readpipe(py2m));

%% initialize solver
msg = struct('solver',solver_name,'config',solver_config,'return_call_log',options.return_call_log);
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
    msg = struct('value', f(reply));
    pipe_send(msg);
end

if isfield(reply, 'error_msg')
    error(['Something borked: ', reply.error_msg]);
end

solution = reply.solution;
details = reply;

end