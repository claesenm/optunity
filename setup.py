from distutils.core import setup

setup(
    name='Optunity',
    version='0.1.0',
    author='Marc Claesen',
    author_email='marc.claesen@esat.kuleuven.be',
    packages=['optunity', 'optunity.test'],
    scripts=[],
    url='https://github.com/claesenm/optunity',
    license='LICENSE.txt',
    description='Optimization routines for hyperparameter tuning.',
    long_description=open('README.rst').read(),
)
