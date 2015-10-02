Julia
=======

.. include:: /global.rst
.. highlight:: julia

In this page we briefly discuss the Julia wrapper, which provides most of Optunity's functionality. 
For a general overview, we recommend reading the :doc:`/user/index`.

For installation instructions, please refer to |installation| and in particular to :ref:`install-julia`. 


Manual
--------

For Julia the following main functions are available:

.. function:: minimize(f[,named_args])

	Perform minimization of the function ``f`` based on the provided ``named_args`` parameters. 

	:param f: minimized function (can be an anonymous or any multi-argument function, or a function accepting ``Dict``)
	:param named_args: various arguments can be provided, such as ``num_evals``, ``solver_name``, but mandatory named arguments are related to the hard constraint on the domain of the function ``f`` (for instance ``x=[-5,5]``). 
	
	:return: ``vars::Dict``, ``details::Dict``
	
	If ``f`` is provided in the anonymous or multi-argument form then the order of the provided hard constraints matter. For instance if one defines::
	
		f = (x,y,z) -> (x-1)^2 + (y-2)^2 + (z+3)^4
	
	or::
	
		f(x,y,z) = (x-1)^2 + (y-2)^2 + (z+3)^4
		
	then we strictly require to provide among other named arguments the following hard constraints::
	
		x=[lb,ub], y=[lb,ub], z=[lb,ub]
		
	If one provides a function accepting ``Dict`` then the order of the provided hard constraints does not matter.
		
	
.. function:: maximize(f[,named_args])

	Perform maximization of the function ``f`` based on the provided ``named_args`` parameters. 

	:param f: minimized function (can be an anonymous or any multi-argument function, or a function accepting ``Dict``)
	:param named_args: various arguments can be provided, such as ``num_evals``, ``solver_name``, but mandatory named arguments are related to the hard constraint on the domain of the function ``f`` (for instance ``x=[-5,5]``). 
	
	

Examples
--------
	
.. code-block:: julia

   using Base.Test

   vars, details = minimize((x,y,z) -> (x-1)^2 + (y-2)^2 + (z+3)^4, x=[-5,5], y=[-5,5], z=[-5,5])
  
   @test_approx_eq_eps vars["x"]  1.0 1.
   @test_approx_eq_eps vars["y"]  2.0 1.
   @test_approx_eq_eps vars["z"] -3.0 1.
   
   testit(x,y,z) = (x-1)^2 + (y-2)^2 + (z+3)^4
   
   vars, details = minimize(testit, num_evals=10000, solver_name="grid search", x=[-5,5], y=[-5,5], z=[-5,5])
   
   @test_approx_eq_eps vars["x"]  1.0 .2
   @test_approx_eq_eps vars["y"]  2.0 .2
   @test_approx_eq_eps vars["z"] -3.0 .2
   
   testit_dict(d::Dict) = -(d[:x]-1)^2 - (d[:y]-2)^2 - (d[:z]+3)^4
   
   vars, details = maximize(testit_dict, num_evals=10000, z=[-5,5], y=[-5,5], x=[-5,5])

   @test_approx_eq_eps vars["x"]  1.0 .2
   @test_approx_eq_eps vars["y"]  2.0 .2
   @test_approx_eq_eps vars["z"] -3.0 .2