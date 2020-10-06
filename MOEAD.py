###########
#  import part #
###########

import numpy as np
import yaml
import random
from scipy.special import comb


from WeightVector import SLD, AWA_transform, determine_neighbor
from Population import init_latin_Hypercube, init_pop, init_ref_point, update_ref_point
from Factory import Problem
from GeneticOperator import variation
from Archive import init_archive, update_archive
from Decomposition import Tchebychev
from AddVector import add_vector, add_vector_random
from DeleteVector import delete_vector_random
from OnlineStoppingCriteria import CR


################
# parameter settings #
################
with open('config.yml') as file:
    params = yaml.safe_load(file)   

###########
#    initialize   #
###########

n_objs = params['n_objs'] #set number of objectives
n_vars = params['n_vars'] #set number of variables

prob_name = params['prob_name'] # set optimization problem
problem = Problem(prob_name, n_vars, n_objs)

H = params['H'] #set number of partitions
T = max(2,int(comb(n_objs+H-1, n_objs-1)//10)) # set neighbor size
delta = params['delta'] # set probability to select parent from neighbor
nr = params['nr']  # set maximum update counts for one offspring 

n_eval = params['n_eval'] # set maximum number of evaluation

ratio = params['ratio'] # set adaption ratio

OSC = CR() # set criteria for vector adaption

#############
# start program #
#############

W = SLD(H, n_objs) # generate a set of weight vectors
B = determine_neighbor(W, T) # determine neighbor
n_pop = len(W) # compute population size

X = init_pop(len(W), n_vars) # initialize a population
Y = problem.do(X) # evaluate fitness

z = init_ref_point(Y) # determine a reference point

archive = init_archive([[Y[i],X[i]] for i in range(n_pop)]) # initialize unbounded external archive

n_fe = len(W)
n_gen = 1

history_all = []

while n_fe < n_eval: 
    n_pop = len(W) # compute population size
    for i in range(n_pop):
        pool = []
        if random.uniform(0.0,1.0) < delta: # determine selection pool by probability
            pool = B[i][:] # neighbor as the pool
        else:
            pool = [j for j in range(n_pop)] # population as the pool
            pool.remove(i)

        xi_ = variation(pool, X, i) # Differential Evolution
        yi_ = problem.do(xi_) # evaluate offspring
        n_fe += 1
        
        z = update_ref_point(z, yi_) # update reference point
        archive = update_archive(archive, [yi_, xi_]) # update unbounded external archive
        
        history_all.append([yi_, n_fe, len(W)]) # record objective values and decision variables
        
        nc = 0 # initialize the update counter
        while True:
            j = np.random.choice(pool) # select a random individual from pool
            if Tchebychev(yi_, W[j], z) < Tchebychev(Y[j], W[j], z): # compare tchebycheff cost of offspring and parent
                Y[j] = yi_ # update parent
                X[j] = xi_
                nc += 1 # cumulate the counter
            pool.remove(j) # remove individual from pool
            if nc == nr or len(pool)==0: # break if counter arrive the upper limit or there are no individual in pool
                break
                
    
    if OSC.OnlineStoppingCriteria(Y): # if Convegence is detected
        p = n_fe/n_eval # compute p 
        if np.random.random() < p: # determine algorithm by probability p
            X, Y, W, T = add_vector(X, Y, W, z, archive,add_ratio = ratio)  # add vector by modified AWA method
        else:  
            X, Y, W, T = add_vector_random(X, Y, W, z, archive,add_ratio = ratio) # add vector randomly
        B = determine_neighbor(W,T) # re-compute neighbor
    else: # if Convergence is not detected
        X, Y, W, T = delete_vector_random(X, Y, W, z, archive) # delete vector randomly
        B = determine_neighbor(W,T) #  re-compute neighbor
        
    n_gen+=1
    
L = int(n_eval//1000)
history_all = history_all[:n_eval]
            
            
        
            













