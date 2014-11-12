classdef Solver
   % SOLVER Handle for an Optunity solver.
   % We recommend constructing Solver objects via optunity.make_solver().
   % Please refer to the documentation thereof for further details.
   %
   % Solver objects have a maximize(f, varargin) function, offering
   % equivalent functionality to optunity.maximize(solver, f, varargin).
   % Please refer to the documentation of optunity.maximize() for details.
   properties (SetAccess = immutable)
      name = '';
      config = struct();
   end % properties
   methods
       function obj = Solver(name, cfg)
          % Create a Solver object. No sanity checks are performed here.
          % The recommended approach is using optunity.make_solver().
          obj.name = name;
          obj.config = cfg;
       end % Solver constructor
       function [solution, details] = maximize(obj, f, varargin)
           % Maximizes f using this solver. Equivalent to:
           % Please refer to optunity.optimize for more details.
           [solution, details] = optunity.optimize(obj, f, maximize, 'true', varargin{:}); 
       end % maximize
       function [solution, details] = minimize(obj, f, varargin)
           % Minimizes f using this solver. Equivalent to:
           % Please refer to optunity.optimize for more details.
           [solution, details] = optunity.optimize(obj, f, maximize, 'false', varargin{:}); 
       end % minimize
       function [solution, details] = optimize(obj, f, varargin)
           % Optimize f using this solver. Equivalent to:
           % Please refer to optunity.optimize for more details.
           [solution, details] = optunity.optimize(obj, f, varargin{:}); 
       end % optimize
       function result = toStruct(obj)
           % Returns a struct representation of the Solver.
           result = obj.config;
           result.solver_name = obj.name;
       end % toStruct
   end % methods
end % Solver class