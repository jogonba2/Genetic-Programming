#!/usr/bin/env python
# -*- coding: utf-8 -*-

from GeneticProgramming import GeneticProgramming
from sklearn.metrics import mean_squared_error
import deap.gp as gp

class GeneticClassifier(GeneticProgramming):
    
    def _f_eval(self,individual):
	f     = gp.compile(individual,self.primitive_set)
	err   = 0
	for i in xrange(len(self.X_train)):
	    if int(eval("f("+",".join(map(str,self.X_train[i].tolist()))+")")) != self.Y_train[i]: err += 1
	return (err,)
	
    def predict(self,X_test):
	assert self.trained==True
	f     = gp.compile(self.hof[0],self.primitive_set)
	Y_res = []
	for i in xrange(len(X_test)):
	    Y_res.append(int(eval("f("+",".join(map(str,X_test[i].tolist()))+")")))
	return Y_res
