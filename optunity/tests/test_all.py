#!/usr/bin/env python

import unittest
import doctest

modules = ['cross_validation', 'functions', 'solvers', 'communication',
           'solvers.GridSearch', 'solvers.RandomSearch', 'solvers.ParticleSwarm',
           'solvers.CMAES', 'solvers.NelderMead']

def load_tests(loader, tests, ignore):
    for mod in modules:
        tests.addTests(doctest.DocTestSuite("optunity." + mod))
    return tests

if __name__ == '__main__':
    unittest.main()
