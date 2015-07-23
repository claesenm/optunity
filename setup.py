from distutils.core import setup
import setuptools

setup(
    name = 'Optunity',
    version = '1.1.1',
    author = 'Marc Claesen',
    author_email = 'marc.claesen@esat.kuleuven.be',
    packages = ['optunity', 'optunity.tests', 'optunity.solvers'],
    scripts = [],
    url = 'http://www.optunity.net',
    license = 'LICENSE.txt',
    description = 'Optimization routines for hyperparameter tuning.',
    long_description = open('README.rst').read(),
    classifiers = ['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Education',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python',
                   'License :: OSI Approved :: BSD License',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.2',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4'
                   ],
    platforms = ['any'],
    keywords = ['machine learning', 'parameter tuning',
                'hyperparameter optimization', 'meta-optimization',
                'direct search', 'model selection', 'particle swarm optimization'
                ],
)
