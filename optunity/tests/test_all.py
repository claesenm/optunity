#!/usr/bin/env python

import unittest
import doctest
import optunity.cross_validation
import optunity.functions

modules = ['cross_validation', 'functions', 'solvers', 'communication']

def load_tests(loader, tests, ignore):
    for mod in modules:
        tests.addTests(doctest.DocTestSuite("optunity." + mod))
    return tests

if __name__ == '__main__':
    unittest.main()
