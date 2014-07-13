classdef Solver
   % SOLVER Handle for an Optunity solver.
   % We recommend constructing Solver objects via optunity.make_solver().
   % Please refer to the documentation thereof for further details.
   %
   % Solver objects have a maximize(f, varargin) function, offering
   % equivalent functionality to optunity.maximize(solver, f, varargin).
   % Please refer to the documentation of optunity.maximize() for details.
   properties
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
           % [solution, details] = optunity.maximize(obj, f, varargin);
           % Please refer to optunity.maximize for more details.
           [solution, details] = optunity.maximize(obj, f, varargin{:}); 
       end % maximize
       function result = toStruct(obj)
           % Returns a struct representation of the Solver.
           result = struct('solver', obj.name, 'config', obj.config);  
       end % toStruct
   end % methods
end % Solver class