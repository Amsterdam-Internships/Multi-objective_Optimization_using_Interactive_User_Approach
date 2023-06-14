# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
from scipy.stats import norm



def weak_pareto_dominates(vec1, vec2):
    """
    Returns whether vec1 weakly dominates vec2
    """
    for i in range(len(vec1)):
        if vec1[i] < vec2[i]:
            return False
    return True


def remove_weak_dominated_by(vector, vv_list):
    """
    Returns a new list of vectors which contains all vectors from vv_set that
    are not weak Pareto-dominated by 'vector'.
    """
    result = []
    for i in range(len(vv_list)):
        if not weak_pareto_dominates(vector, vv_list[i]):
            result.append(vv_list[i])
    return result



def pareto_prune(vv_set):
    """
    Returns a new list of value vectors from which all Pareto-dominated vectors
    have been removed.
    For Pseudo-code, see e.g., PPrune (Algorithm 2, page 34, Chapter 3) from
        Diederik M. Roijers - Multi-Objective Decision-Theoretic Planning,
        PhD Thesis, University of Amsterdam, 2016.
    """
    V = vv_set.copy()
    result = []
    while len(V):
        current_non_dominated = V[0]
        for i in range(len(V)):
            if weak_pareto_dominates(V[i], current_non_dominated):
                current_non_dominated = V[i]
        result.append(current_non_dominated)
        V = remove_weak_dominated_by(current_non_dominated, V)
    return result

##########################
numpoints = 100
pointset = [] 
numobjectives = 2

while len(pointset)<numpoints :
    vec = norm.rvs(size=numobjectives) #Gaussian to ensure uniformity over the hypersphere
    vec2 = map(lambda x: x*x, vec)
    normaliser =  np.sqrt(sum(vec2)) #to project onto the hypersphere
    vecnorm = list(map(lambda x: abs(x)/normaliser, vec)) #abs to project onto the positive part of the hypersphere
    pointset.append(vecnorm)

#print(list(map(lambda y: sum(list(map(lambda x: x*x, y))), pointset)) ) #checking norm is 1
#print(len(pareto_prune(pointset))) #check all undominated

np.savetxt("synthetic_pcs_sets/obj2size100.csv", np.array(pointset), delimiter=",")    
