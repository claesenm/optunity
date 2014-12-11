function [solution, details] = maximize(obj, f, varargin)
    % Maximizes f using this solver. Equivalent to:
    % Please refer to optimize for more details.
    [solution, details] = optimize(obj, f, maximize, 'true', varargin{:}); 
end % maximize
