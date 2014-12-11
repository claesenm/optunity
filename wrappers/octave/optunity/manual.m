function [solver_names] = manual( varargin )
%MANUAL Prints the manual of specified solver or Optunity in general.
%
% If no solver name is specified, a general manual is printed along with
% a list of available solvers. If a name is specified, the same name is
% returned.

[sock, pid, cleaner] = comm_launch();

if nargin > 0
    init = struct('manual', varargin{1});
else
    init = struct('manual','');
end

json_request = comm_json_encode(init);
comm_writepipe(sock, json_request);

json_reply = comm_readpipe(sock);
reply = comm_json_decode(json_reply);

if isfield(reply, 'error_msg')
   error(['Error retrieving manual: ',reply.error_msg]);
end

for ii=1:numel(reply.manual)
    disp(reply.manual{ii});
end

solver_names = reply.solver_names;
end
