import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy as np

def calc_SL(W,n_objs=3):
    num = min(n_objs,len(W))

    neigh = NearestNeighbors(num)
    neigh.fit(W)

    neighbors = neigh.kneighbors(W, num)[0].tolist()
    SL = [np.prod(item[1:num]) for item in neighbors]
    return SL

def delete_vector_random(X, Y, W, z, Archive, delete_ratio = 0.01,lower_pop=10):
    arg_edges = (W == 1).any(axis=1)
    W_edges = W[arg_edges]
    X_edges = X[arg_edges]
    Y_edges = Y[arg_edges]
    if len(W)<=lower_pop:
        return X, Y, W, max(2, len(W)//10)
    
    W_ = W[~arg_edges]
    X_ = X[~arg_edges]
    Y_ = Y[~arg_edges]
    
    num  = int(len(W_) - len(W_)*delete_ratio)
    arg_remain = np.random.choice(range(len(W_)),num,replace=False)
    W_ = W_[arg_remain]
    X_ = X_[arg_remain]
    Y_ = Y_[arg_remain]
    
    W_ = np.append(W_, W_edges,axis=0)
    X_ = np.append(X_, X_edges,axis=0)
    Y_ = np.append(Y_, Y_edges,axis=0)
    T = max(2, len(W_)//10)
    
    return X_, Y_, W_, T


# def delete_vector(X, Y, W, z, Archive, delete_ratio = 0.01,lower_pop=10):
#     arg_edges = (W == 1).any(axis=1)
#     W_edges = W[arg_edges]
#     X_edges = X[arg_edges]
#     Y_edges = Y[arg_edges]
#     if len(W)<=lower_pop:
#         return X, Y, W, max(2, len(W)//10)
    
#     W_ = W[~arg_edges]
#     X_ = X[~arg_edges]
#     Y_ = Y[~arg_edges]
    
#     SL = calc_SL(W_)
#     num  = int(len(W_) - len(W_)*delete_ratio)
#     L = np.argsort(SL)[::-1][-num:]
        
#     arg_remain = np.array([False if i in L else True for i in range(len(W_))])

#     W_ = W_[arg_remain]
#     X_ = X_[arg_remain]
#     Y_ = Y_[arg_remain]
    
#     W_ = np.append(W_, W_edges,axis=0)
#     X_ = np.append(X_, X_edges,axis=0)
#     Y_ = np.append(Y_, Y_edges,axis=0)
#     T = max(2, len(W_)//10)
    
#     return X_, Y_, W_, T

def delete_vector_AWA(X, Y, W, z, Archive, update_ratio = 0.05,first_pop=105):
    arg_edges = (W == 1).any(axis=1)
    W_edges = W[arg_edges]
    X_edges = X[arg_edges]
    Y_edges = Y[arg_edges]
    
    W_ = W[~arg_edges]
    X_ = X[~arg_edges]
    Y_ = Y[~arg_edges]
    
    nus = max(int(first_pop*update_ratio),1)
    L = []
    for i in range(nus):
        SL = calc_SL(W_)
        arg = np.argmin(SL)
        index = np.ones(len(W_),dtype=bool)
        index[arg] = False
        W_ = W_[index]
        X_ = X_[index]
        Y_ = Y_[index]
    
    W_ = np.append(W_, W_edges,axis=0)
    X_ = np.append(X_, X_edges,axis=0)
    Y_ = np.append(Y_, Y_edges,axis=0)
    T = max(2, len(W_)//10)
    
    return X_, Y_, W_, T