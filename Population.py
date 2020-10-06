import numpy as np
from pyDOE import lhs
import random

def init_pop(population,numberOfVariables):
    Individuals = []
    for i in range(population):
        p = []
        for j in range(numberOfVariables):
            p.append(random.uniform(0.0,1.0))
        Individuals.append(p)
    return np.array(Individuals)

def init_latin_Hypercube(population,numberOfVariables):
    return lhs(numberOfVariables,samples=population,criterion='center')


import numpy as np

def init_ref_point(F):
    return np.min(F, axis=0)

def update_ref_point(ref_point, fy):
    tmp = np.vstack([ref_point, fy])
    return np.min(tmp, axis=0)