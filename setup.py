from distutils.core import setup

setup(
    name='Optunity',
    version='0.2.0',
    author='Marc Claesen',
    author_email='marc.claesen@esat.kuleuven.be',
    packages=['optunity', 'optunity.test'],
    scripts=[],
    url='http://optunity.readthedocs.org',
    license='LICENSE.txt',
    description='Optimization routines for hyperparameter tuning.',
    long_description=open('README.rst').read(),
    install_requires=[
          'deap >= 1.0.1',
      ],
)
