function writepipe( m2py, json )
%WRITEPIPE Sends the json string to the SOAP subprocess over its stdin pipe.

m2py.println(java.lang.String(json));
m2py.flush();
end