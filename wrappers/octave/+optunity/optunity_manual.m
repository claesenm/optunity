function [solver_names] = optunity_manual( varargin )
%MANUAL Prints the manual of specified solver or Optunity in general.
%
% If no solver name is specified, a general manual is printed along with
% a list of available solvers. If a name is specified, the same name is
% returned.

[m2py, py2m, stderr, pid, cleaner] = optunity_comm_launch();

if nargin > 0
    init = struct('manual', varargin{1});
else
    init = struct('manual','');
end

json_request = optunity_comm_json_encode(init);
optunity_comm_writepipe(m2py, json_request);

json_reply = optunity_comm_readpipe(py2m);
reply = optunity_comm_json_decode(json_reply);

if isfield(reply, 'error_msg')
   error(['Error retrieving manual: ',reply.error_msg]);
end

for ii=1:numel(reply.manual)
    disp(reply.manual{ii});
end

solver_names = reply.solver_names;
end
