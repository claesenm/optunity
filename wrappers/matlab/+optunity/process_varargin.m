function options = process_varargin(defaults, actual_varargin, varargin)

limit_to_options = true;
if numel(varargin)
    limit_to_options = varargin{1};
end
% inspired by http://stackoverflow.com/a/2776238/2148672

%# read the acceptable names
options = defaults;
optionNames = fieldnames(options);

%# count arguments
nArgs = length(actual_varargin);
if round(nArgs/2)~=nArgs/2
   error('varargin requires propertyName/propertyValue pairs')
end

for pair = reshape(actual_varargin,2,[]) %# pair is {propName;propValue}
   inpName = lower(pair{1}); %# make case insensitive   
   if ~limit_to_options || any(strcmp(inpName,optionNames)) 
      %# overwrite options. If you want you can test for the right class here
      %# Also, if you find out that there is an option you keep getting wrong,
      %# you can use "if strcmp(inpName,'problemOption'),testMore,end"-statements
      options.(inpName) = pair{2};
   else
      error('%s is not a recognized parameter name',inpName)
   end
end