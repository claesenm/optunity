function obj = optunity_Solver(name, cfg)
   % SOLVER Handle for an Optunity solver.
   % We recommend constructing Solver objects via optunity_make_solver().
   % Please refer to the documentation thereof for further details.
   %
   % Solver objects have a maximize(f, varargin) function, offering
   % equivalent functionality to optunity_maximize(solver, f, varargin).
   % Please refer to the documentation of optunity_maximize() for details.

obj = cfg;
obj.solver_name = name;
