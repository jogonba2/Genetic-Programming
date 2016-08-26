#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Operators import Operators
import random
import string
import deap.gp as gp
import deap.creator as crt
import deap.base as bs
import deap.tools as tl
import deap.algorithms as al
import numpy as np

class GeneticProgramming:
    
    def __init__(self,X_train,Y_train,n_constants=1,prob_mating=0.25,prob_mutating=0.1,it=50,tournament_size=3,
		 min_constant=-1,max_constant=1,min_deep=1,max_deep=3,n_population=25,
		 l=Operators()._get_operators(),n_best_classifiers=1,verbose=False):
		     
	self.X_train       		= X_train
	self.Y_train       		= Y_train
	self.n_constants   		= n_constants
	self.min_constant  		= min_constant
	self.max_constant  		= max_constant
	self.min_deep      		= min_deep
	self.max_deep      		= max_deep
	self.prob_mating   		= prob_mating
	self.prob_mutating 		= prob_mutating
	self.it            		= it
	self.vb                         = verbose
	self.tournament_size 		= tournament_size
	self.n_population  		= n_population
	self.n_best_classifiers         = n_best_classifiers
	self.l             		= l
	self.primitive_set 		= self.__create_primitive_set(self.X_train.shape[1])
	self.toolbox       		= self.__config_toolbox()
	self.population    		= self.toolbox.population(n=self.n_population)
	self.hof           		= tl.HallOfFame(self.n_best_classifiers)
	self.stats_fit 			= tl.Statistics(lambda ind: ind.fitness.values)
	self.stats_size 		= tl.Statistics(len)
	self.mstats 			= tl.MultiStatistics(fitness=self.stats_fit, size=self.stats_size)
	self.__config_statistics()
	self.log 			= None
	self.trained                    = False
    
    def __config_statistics(self):
	self.mstats.register("avg", np.mean)
	self.mstats.register("std", np.std)
	self.mstats.register("min", np.min)
	self.mstats.register("max", np.max)
    
    def __create_primitive_set(self,n_vars):
	primitive_set = gp.PrimitiveSet("main_primitive_set",n_vars)
	for i in xrange(len(self.l)): primitive_set.addPrimitive(self.l[i][0],self.l[i][1])
	for i in xrange(self.n_constants): primitive_set.addEphemeralConstant(''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5)),
									      lambda: random.uniform(self.min_constant,self.max_constant))
	return primitive_set
	
    def __config_toolbox(self):
	crt.create("FitnessMin",bs.Fitness,weights=(-1.0,))
	crt.create("Individual",gp.PrimitiveTree,fitness=crt.FitnessMin,pset=self.primitive_set)
	toolbox = bs.Toolbox()
	toolbox.register("expr",gp.genFull,pset=self.primitive_set,min_=self.min_deep,max_=self.max_deep)
	toolbox.register("individual", tl.initIterate, crt.Individual,toolbox.expr)
	toolbox.register("population",tl.initRepeat,list,toolbox.individual)
	toolbox.register("evaluate",self._f_eval)
	toolbox.register("select",tl.selTournament,tournsize=self.tournament_size)
	toolbox.register("mate",gp.cxOnePoint)
	toolbox.register("expr_mut",gp.genFull,min_=self.min_deep,max_=self.max_deep)
	toolbox.register("mutate",gp.mutUniform,expr=toolbox.expr_mut,pset=self.primitive_set)
	return toolbox
    
    def show_logs(self): 
	assert self.trained==True
	print self.log
	
    def get_best_classifiers(self):
	assert self.trained==True
	return self.hof
    
    def fit(self): 
	self.population,self.log = al.eaSimple(self.population,self.toolbox,self.prob_mating,
					       self.prob_mutating,self.it,stats=self.mstats,
					       halloffame=self.hof,verbose=self.vb)
	self.trained = True
