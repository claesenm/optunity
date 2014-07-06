function folds = generate_folds( n, varargin )
%GENERATE_FOLDS Generates k-fold cross-validation folds.
% todo

%% process varargin
defaults = struct('num_instances', n, ...
    'num_folds', 10, ...
    'num_iter', 1, ...
    'strata', NaN, ...
    'clusters', NaN);
options = optunity.process_varargin(defaults, varargin);

if ~iscell(options.strata)
    options = rmfield(options, 'strata');
end
if ~iscell(options.clusters)
    options = rmfield(options, 'clusters');
end

assert(options.num_instances > options.num_folds, ...
    'Number of instances less than number of folds!');

[m2py, py2m, stderr, subprocess, cleaner] = optunity.comm.launch();

init = struct('generate_folds', options);
json_request = optunity.comm.json_encode(init);
optunity.comm.writepipe(m2py, json_request);

json_reply = optunity.comm.readpipe(py2m);
reply = optunity.comm.json_decode(json_reply);

if isfield(reply, 'error_msg')
   error(['Error retrieving manual: ',reply.error_msg]);
end

% Optunity returns 0-based indices, turn into 1-based
% turn into a matrix of folds instead of cells
niter = numel(reply.folds);
folds = zeros(n, niter);
for ii=1:niter
    for jj=1:numel(reply.folds{ii})
        folds(1+reply.folds{ii}{jj}, ii) = jj;
    end
end

end