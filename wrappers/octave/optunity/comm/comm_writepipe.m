function json = comm_writepipe( socket, json )
%WRITEPIPE Sends the json string to the Optunity subprocess over its stdin pipe.

global DEBUG_OPTUNITY
if DEBUG_OPTUNITY
   disp(['Sending ',json]);
   fflush(stdout);
end

send(socket, [json, "\n"]);

end
