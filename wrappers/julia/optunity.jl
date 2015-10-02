using PyCall

@pyimport optunity
include("nogc.jl")

map_args = args -> map(e -> e[1], args)

function wrap_call(f::Function,all_args)
	wrapper_(;kw...) = f(Dict(kw))

	function wrapper(;kw...)
		for e in kw 
			ex = e[1]; @eval $ex = $e[2]
		end
		args = intersect(all_args,map_args(kw))
		f(map(e -> (@eval $e), args)...)
	end

	try
		methods(f).defs.sig <: Tuple{Dict} ? wrapper_ : wrapper
	catch ArgumentError
		wrapper
	end
end

function minimize(f::Function;kws...)
	results = @nogc optunity.minimize(wrap_call(f,map_args(kws));kws...)
	results[1], results[2]
end

function maximize(f::Function;kws...)
	results = @nogc optunity.maximize(wrap_call(f,map_args(kws));kws...)
	results[1], results[2]
end