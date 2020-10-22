import copy
import numpy as np
import random

def DefferentialEvolution(ind1,ind2,ind3,numberOfVariables,maxi=1.0):
    C =  1.0
    F = 0.5
    ita = 20
    ita_pow = 1/(ita+1)
    MIN = 0
    MAX = maxi
    N = numberOfVariables
    MR = 1.0/N
    variables = []
    irand = random.randrange(numberOfVariables)
    #Defferential Evolution
    for i in range(N):
        if(random.uniform(0.0,1.0) < C): 
            tmp = ind1[i]+F*(ind2[i]-ind3[i])
        else:
            tmp = ind1[i]
        if tmp > MAX or tmp < MIN:
            tmp = max(MIN,min(MAX,tmp))
        variables.append(tmp) 
        
    #polynomial mutation
    
    for i in range(N):
        if  random.uniform(0.0,1.0)<MR:
            rnd = random.uniform(0.0,1.0)
            if rnd < 0.5:
                sigma = (2*rnd)**ita_pow-1
            else:
                sigma = 1-(2-2*rnd)**ita_pow
            variables[i] = variables[i]+sigma*(MAX-MIN)
        if variables[i] > MAX  or variables[i]<MIN:
            variables[i] = max(MIN,min(MAX,variables[i]))
    return variables

def variation(pool,Individuals,i):
    new_Individual = copy.deepcopy(Individuals[i])
    index = copy.deepcopy(pool)
    random.shuffle(index)
    x2 = index[0]
    x3 = index[1]
    new_Individual = DefferentialEvolution(Individuals[i],Individuals[x2],Individuals[x3],len(Individuals[i]))
    return new_Individual