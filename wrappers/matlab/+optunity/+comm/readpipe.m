function [ content ] = readpipe( pipe )
%READPIPE Reads data chunks from Optunity over its stdout pipe.

content = char(pipe.readLine());

global DEBUG_OPTUNITY
if DEBUG_OPTUNITY
   disp(['Received ',content]);
end

if numel(content) == 0
    error('Broken pipe: did not receive any data.');
end
end
