function [ json ] = comm_json_encode( struct )
%JSON_ENCODE Encodes the given struct in json format.

if isstruct(struct)
    fields = fieldnames(struct);
    strings = arrayfun(@(x) ['"',fields{x},'": ', ...
        comm_json_encode(struct.(fields{x}))], ...
        1:numel(fields),'UniformOutput',false);
    json = ['{',strjoin(strings,', '),'}'];
    
elseif ischar(struct)
    json = ['"',struct,'"'];
    
elseif iscell(struct) && isempty(struct)
    json = '[]';
    
elseif iscell(struct) && numel(struct) == 1 % dealing with a 1-element cell
    string = comm_json_encode(struct{1});    
    json = ['[',string,']'];
   
elseif numel(struct) > 1
    if iscell(struct)
        strings = cellfun(@(x) comm_json_encode(x), struct,'UniformOutput',false);
        json = ['[', strjoin(strings, ', '), ']'];
    else
        strings = arrayfun(@(x) comm_json_encode(x), struct,'UniformOutput',false);
        json = ['[', strjoin(strings,', '),']'];
    end
   
elseif isnumeric(struct)
    json = num2str(struct);
    
elseif islogical(struct)
    if struct
        json = 'true';
    else
        json = 'false';
    end 
elseif iscell(struct)
    strs = cellfun(@(x) comm_json_encode(x), struct, 'UniformOutput', false);
    json = strjoin(strs, ', ');
else
    error('UNKNOWN DATA');
end
end


function str = strjoin(data, delim)
if iscell(data)
    str = cellfun(@(x) [x, delim], data, 'UniformOutput', false);
else
    error('NOT IMPLEMENTED!');
    %     str = arrayfun(@(x) [x, delim], data, 'UniformOutput', false);
end
% str
cat = str{1};
for ii=2:numel(str)
    cat = [cat, str{ii}];
end
% str = cell2mat(str);
str = cat(1:end-numel(delim));
end
