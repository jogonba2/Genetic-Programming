#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris,load_boston,load_diabetes
from GeneticClassifier import GeneticClassifier
from GeneticRegressor  import GeneticRegressor

if __name__ == "__main__":
    
    ## Load dataset ##
    iris_data = load_iris()
    X = iris_data.data[:, :] 
    Y = iris_data.target
    X_train,X_test = X[0:120],X[30:]
    Y_train,Y_test = Y[0:120],Y[30:]
    ##################
    
    ## Classification ##
    gc = GeneticClassifier(X_train,Y_train)
    gc.fit()
    print "Classifier: ",gc.get_best_classifiers()[0]
    print "Error: ",(float((gc.predict(X_test)!=Y_test).sum())/len(Y_test))*100,"%"
    #print gc.show_logs()
    ####################
    
    ## Regression ##
    gr = GeneticRegressor(X_train,Y_train)
    gr.fit()
    print "Regressor: ",gr.get_best_classifiers()[0]
    print gr.predict(X_test)
    print Y_test[30:]
    #print gc.show_logs()
    ################
    
