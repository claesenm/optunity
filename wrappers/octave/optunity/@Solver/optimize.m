function [solution, details] = optimize(obj, f, varargin)
    % Optimize f using this solver. Equivalent to:
    % Please refer to optimize for more details.
    [solution, details] = optimize(toStruct(obj), f, varargin{:}); 
end % optimize
