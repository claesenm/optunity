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
    classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: BSD License'
    'Topic :: Scientific/Engineering :: Artificial Intelligence'
    'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    platforms=['any'],
    keywords=['machine learning', 'parameter tuning',
              'hyperparameter optimization', 'meta-optimization',
              'direct search', 'model selection'
    ],
)
