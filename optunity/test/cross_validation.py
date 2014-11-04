#! /usr/bin/env python

import doctest
from optunity.cross_validation import *

doctest.testfile('cross_validation.py', package='optunity', globs=locals())
