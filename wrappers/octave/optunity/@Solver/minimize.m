function [solution, details] = minimize(obj, f, varargin)
    % Minimizes f using this solver. Equivalent to:
    % Please refer to optimize for more details.
    [solution, details] = optimize(obj, f, maximize, 'false', varargin{:}); 
end % minimize
