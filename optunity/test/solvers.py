#! /usr/bin/env python

import math
import doctest
from optunity.solvers import *

doctest.testfile('solvers.py', package='optunity', globs=locals())
