function json = writepipe( m2py, json )
%WRITEPIPE Sends the json string to the Optunity subprocess over its stdin pipe.

global DEBUG_OPTUNITY
if DEBUG_OPTUNITY
   disp(['Sending ',json]);
end

m2py.println(java.lang.String(json));
m2py.flush();
end
