tests = ["test1", "test2"]

print_with_color(:blue, "Running tests:\n")

for t in tests
	test_fn = "$t.jl"
	print_with_color(:green, "* $test_fn\n")
	include(test_fn)
end