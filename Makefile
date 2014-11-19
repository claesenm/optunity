tests: 
	nosetests optunity/solvers/ -i optunity/*.py -e optunity/piped.py --with-doctest -v

sphinx:
	sphinx-apidoc --separate -o docs/api/ optunity/
