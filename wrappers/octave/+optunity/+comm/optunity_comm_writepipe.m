function json = optunity_comm_writepipe( m2py, json )
%WRITEPIPE Sends the json string to the Optunity subprocess over its stdin pipe.

global DEBUG_OPTUNITY
if DEBUG_OPTUNITY
   disp(['Sending ',json]);
end

%fputs(m2py, [json, "\n"]);
fdisp(m2py, json)
r = fflush(m2py);

if r ~= 0
    error('Unable to flush pipe.')
end

end
