from distutils.core import setup

setup(
    name='Optunity',
    version='0.1.0',
    author='Marc Claesen',
    author_email='marc.claesen@esat.kuleuven.be',
    packages=['optunity'],
    scripts=['bin/blah'],
    url='http://pypi.python.org/pypi/Optunity/',
    license='LICENSE.txt',
    description='Optimization routines for hyperparameter tuning.',
    long_description=open('README.txt').read(),
    install_requires=[
    ],
)
