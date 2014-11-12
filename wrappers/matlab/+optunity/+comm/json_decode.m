function [ struct, stop ] = json_decode( json )
%JSON_DECODE Decodes given JSON string.

start = 1;

if json(start) == '{'
    [struct, stop] = decode_dict(json(start:end));
elseif json(start) == '['
    [struct, stop] = decode_list(json(start:end));
elseif json(start) == '"'
    % string
    stop = find(json(start+1:end) == '"', 1) + 1;
    struct = json(start+1:start+stop-2);
else
    % numeric
    stop = find(json(start:end) == ',' ...
        | json(start:end) == '}' ...
        | json(start:end) == ']', 1)-1;
    struct = str2double(json(start:start+stop-1));
end

stop = stop + start - 1;
end

function [result, stop] = decode_dict(json)
result = struct();
stop = 2;
while json(stop) ~= '}'
    [key, value, stop] = decode_kv(json, stop);
    result.(key) = value;
    stop = stop + 1;
    if json(stop) == ','
        stop = stop+2;
    end
end

end

function [key, value, stop] = decode_kv(json, start)
% key is always a string, find its closing "
stop = find(json(start+1:end) == '"', 1);
key = json(start+1:start+stop-1);
stop = start + stop;

assert(strcmp(json(stop:stop+2),'": '), 'ILLEGAL JSON FORMAT');
stop = stop+3; % skipping '": '

[value, offset] = optunity.comm.json_decode(json(stop:end));
stop = stop + offset - 1;
end

function [list, stop] = decode_list(json)
stop = 2;

use_cell = false;
if json(stop) == '"'
    use_cell = true;
elseif json(stop) == '{'
    use_cell = true;
elseif json(stop) == '['
    use_cell = true;
end

list = [];
while json(stop) ~= ']'
    [value, offset] = optunity.comm.json_decode(json(stop:end));
    if use_cell
        list{end+1} = value;
    else
        list(end+1) = value;
    end
    stop = offset + stop;
    if json(stop) == ','
        stop = stop+2;
    end
end

if use_cell
    list = transpose(list);
end
end