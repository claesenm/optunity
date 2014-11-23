function folds = generate_folds( n, varargin )
%GENERATE_FOLDS Generates k-fold cross-validation folds.
%
%- n: the number of instances in the data set
%- varargin: a list of optional key:value pairs to further configure
%      the cross-validation procedure
%  - num_folds: number of folds to use in cross-validation
%      default: 10
%  - num_iter: number of cross-validation iterations to perform 
%      default: 1
%  - strata: cell array containing strata, e.g. indices of instances that
%      must be spread out across folds (default: empty)
%  - clusters: cell array containing clusters, e.g. indices of instances
%      that must be kept within a single fold
%      default: empty

%% process varargin
defaults = struct('num_instances', n, ...
    'num_folds', 10, ...
    'num_iter', 1, ...
    'strata', [], ...
    'clusters', []);
options = optunity.process_varargin(defaults, varargin);

if isempty(options.strata)
    options = rmfield(options, 'strata');
end
if isempty(options.clusters)
    options = rmfield(options, 'clusters');
end

assert(options.num_instances >= options.num_folds, ...
    'Number of instances less than number of folds!');

[m2py, py2m, ~, ~, cleaner] = optunity.comm.launch();

init = struct('generate_folds', options);
json_request = optunity.comm.json_encode(init);
optunity.comm.writepipe(m2py, json_request);

json_reply = optunity.comm.readpipe(py2m);
reply = optunity.comm.json_decode(json_reply);

if isfield(reply, 'error_msg')
   error(['Error generating folds: ',reply.error_msg]);
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