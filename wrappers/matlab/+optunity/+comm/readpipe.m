function [ content ] = readpipe( pipe )
%READPIPE Reads data chunks from Optunity over its stdout pipe.

content = char(pipe.readLine());
if numel(content) == 0
    error('Broken pipe: did not receive any data.');
end
end
