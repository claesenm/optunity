using PyCall

@pyimport optunity

map_args = args -> map(e -> e[1], args)

function wrap_call(f::Function,all_args)
	function wrapper(;kw...)
		for e in kw 
			ex = e[1]; @eval $ex = $e[2]
		end
		args = intersect(all_args,map_args(kw))
		f(map(e -> (@eval $e), args)...)
	end
	function wrapper2(;kw...)
		f(Dict(kw))
	end

	try
		methods(f).defs.sig <: Tuple{Dict} ? wrapper2 : wrapper
	catch ArgumentError
		wrapper
	end
end

function minimize(f::Function;kws...)
	optunity.minimize(wrap_call(f,map_args(kws));kws...)
end

function maximize(f::Function;kws...)
	optunity.maximize(wrap_call(f,map_args(kws));kws...)
end