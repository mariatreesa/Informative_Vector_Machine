# The Informative Vector Machine Algorithm
 
 This package implements the Informative Vector Machine algorithm discussed in the 
 paper http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.2.4533&rep=rep1&type=pdf
 
 This is a framework for Sparse Gaussian Process Regression. This algorithm returns the indexes
 of an active subset (d<<n) of datapoints which contains more information. This method can 
 significantly increase the training speed for Gaussian Process regression.
 
 The selection of datapoints is based on Information theory, where the predictors which minimizes
 the entropy of posterior process is greedily selected in an iterative fashion
