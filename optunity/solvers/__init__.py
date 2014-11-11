#! /usr/bin/env python

# Copyright (c) 2014 KU Leuven, ESAT-STADIUS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither name of copyright holders nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Module to take care of registering solvers for use in the main Optunity API.

Main classes in this module:

* :class:`Solver`
* :class:`GridSearch`
* :class:`RandomSearch`
* :class:`NelderMead`
* :class:`ParticleSwarm`
* :class:`CMA_ES`
  :class:`CSA`

.. warning::
    :class:`CMA_ES` require DEAP_.

    .. _DEAP: https://code.google.com/p/deap/


Bibliographic references for some solvers:

.. [HANSEN2001] Nikolaus Hansen and Andreas Ostermeier. *Completely
    derandomized self-adaptation in evolution  strategies*.
    Evolutionary computation, 9(2):159-195, 2001.

.. [DEAP2012] Felix-Antoine Fortin, Francois-Michel De Rainville, Marc-Andre Gardner,
    Marc Parizeau and Christian Gagne, *DEAP: Evolutionary Algorithms Made Easy*,
    Journal of Machine Learning Research, pp. 2171-2175, no 13, jul 2012.


.. moduleauthor:: Marc Claesen

"""

from .GridSearch import GridSearch
from .RandomSearch import RandomSearch
from .NelderMead import NelderMead
from .ParticleSwarm import ParticleSwarm
from .CMAES import CMA_ES
