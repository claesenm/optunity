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

import itertools
import functools
import multiprocessing
import threading
import copy


__all__ = ['pmap', 'Future']


def pmap(f, *args):
    pool = multiprocessing.Pool()
    result = pool.map(f, *args)
    return result

# http://code.activestate.com/recipes/84317-easy-threading-with-futures/
class Future:
    def __init__(self,func,*param):
        # Constructor
        self.__done=0
        self.__result=None
        self.__status='working'

        self.__S=threading.Semaphore(0)

        # Run the actual function in a separate thread
        self.__T=threading.Thread(target=self.Wrapper, args=(func, param))
        self.__T.setName("FutureThread")
        self.__T.daemon=True
        self.__T.start()

    def __repr__(self):
        return '<Future at '+hex(id(self))+':'+self.__status+'>'

    def __call__(self):
        try:
            self.__S.acquire()
            # We deepcopy __result to prevent accidental tampering with it.
            a=copy.deepcopy(self.__result)
        finally:
            self.__S.release()
        return a

    def join(self):
        self.__T.join()

    def Wrapper(self, func, param):
        # Run the actual function, and let us housekeep around it
#        try:
        self.__result=func(*param)
#        except:
#            self.__result="Exception raised within Future"
        self.__status=`self.__result`
        self.__S.release()


if __name__ == '__main__':
    pass
