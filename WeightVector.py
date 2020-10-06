import numpy as np
import math
import itertools
from sklearn.neighbors import NearestNeighbors


def SLD(H,numberOfObjective):
    n = numberOfObjective
    lis = []
    for i in range(H):
        lis.append(i/H)
    lis.append(1)
    ans = []
    for i in itertools.product(lis, repeat=n):
        if sum(i) == 1:
            ans.append(list(i))
    return np.array(ans)

def AWA_transform(H, n_objs):
    W = SLD(H, n_objs)
    small = 1/(10**9+7)
    for i in range(len(W)):
        if np.sum(W==0)==(n_objs-1):
            continue
        elif np.sum(W==0)>=1:
            idx = np.where(W==0)
            W[idx] = W[idx]+small
        W = 1/W

    for i in range(len(W)):
        W[i] = W[i]/sum(W[i])
    return W


def random_weights(population_size, nobjs):    
    weights = []
    
    if nobjs == 2:
        weights = [[1, 0], [0, 1]]
        weights.extend([(i/(population_size-1.0), 1.0-i/(population_size-1.0)) for i in range(1, population_size-1)])
    else:
        # generate candidate weights
        candidate_weights = []
        
        for i in range(population_size*50):
            random_values = [np.random.random() for _ in range(nobjs)]
            candidate_weights.append([x/sum(random_values) for x in random_values])
        
        # add weights for the corners
        for i in range(nobjs):
            weights.append([0]*i + [1] + [0]*(nobjs-i-1))
            
        # iteratively fill in the remaining weights by finding the candidate
        # weight with the largest distance from the assigned weights
        while len(weights) < population_size:
            max_index = -1
            max_distance = -np.inf
            
            for i in range(len(candidate_weights)):
                distance = np.inf
                
                for j in range(len(weights)):
                    temp = np.sqrt(sum([(candidate_weights[i][k]-weights[j][k])**2 for k in range(nobjs)]))
                    distance = min(distance, temp)
                    
                if distance > max_distance:
                    max_index = i
                    max_distance = distance
                    
            weights.append(candidate_weights[max_index])
            del candidate_weights[max_index]   
    return np.array(weights)

def determine_neighbor(Vector,numberOfNeighbor):
    neigh = NearestNeighbors(numberOfNeighbor)
    neigh.fit(Vector)
    neighbors = neigh.kneighbors(Vector, numberOfNeighbor)
    neighbor = [item.tolist() for item in neighbors[1]]
    return neighbor

