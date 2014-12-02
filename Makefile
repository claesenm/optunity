tests: 
	nosetests optunity/solvers/ -i optunity/*.py -e optunity/standalone.py --with-doctest -v

sphinx:
	sphinx-apidoc --separate -o docs/api/ optunity/

msi:
	python setup.py bdist --formats=msi

wininst:
	python setup.py bdist --formats=wininst
