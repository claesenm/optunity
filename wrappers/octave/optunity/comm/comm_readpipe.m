function [ content ] = comm_readpipe( socket )
%READPIPE Reads data chunks from Optunity over its stdout pipe.

content = '';
while true
    c = char(recv(socket, 1));
    if c == "\n"
        break
    end
    content = [content, c];
end

global DEBUG_OPTUNITY
if DEBUG_OPTUNITY
   disp(['Received ',content]);
   fflush(stdout);
end

if numel(content) == 0
    error('Broken pipe: did not receive any data.');
end
end
