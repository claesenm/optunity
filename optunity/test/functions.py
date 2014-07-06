#! /usr/bin/env python

import math
import doctest
from optunity.functions import *

doctest.testfile('functions.py', package='optunity', globs=locals())
