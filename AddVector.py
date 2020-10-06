import numpy as np
from sklearn.neighbors import NearestNeighbors

from Decomposition import Tchebychev

def calc_SL(Archive,n_objs):
    objs = [item[0] for item in Archive]
    num = min(n_objs,len(Archive))

    neigh = NearestNeighbors(num)
    neigh.fit(objs)

    neighbors = neigh.kneighbors(objs, num)[0].tolist()
    SL = [np.prod(item[1:num]) for item in neighbors]
    return SL

def add_vector_AWA(X, Y, W, z, Archive, add_ratio = 0.05,epsilon = 10**-7,first_pop=105):
    W_add = []
    Y_add = []
    X_add = []
    Y_ = [item[0] for item in Archive]
    X_= [item[1] for item in Archive]
    SL = calc_SL(Archive,len(z))
    SL_list = [True for i in range(len(SL))]
    nus = max(int(first_pop*add_ratio),1)
    for count in range(nus):
#         Y_sp = Y_[np.argmax(SL)]
        Y_sp = Y_[np.argmax(np.array(SL)[SL_list])]
        Y_sp_ideal = sum([1/(Y_sp[i] -z[i]+epsilon) for i in range(len(z))])
        X_add.append(X_[np.argmax(SL)])
        Y_add.append(Y_sp)
        W_add.append([(1/(Y_sp[i] - z[i]+epsilon))/Y_sp_ideal for i in range(len(z))])
#         SL.remove(max(SL))
        SL_list[np.argmax(np.array(SL)[SL_list])] = False
    X = np.append(X, np.array(X_add),axis=0)
    Y = np.append(Y, np.array(Y_add),axis=0)
    W = np.append(W, np.array(W_add),axis=0)
    T = max(2, len(W)//10)
    
    return X, Y, W, T

# def add_vector(X, Y, W, z, Archive, add_ratio = 0.03,epsilon = 10**-7):
#     W_add = []
#     Y_add = []
#     X_add = []
#     Y_ = [item[0] for item in Archive]
#     X_= [item[1] for item in Archive]
#     SL = calc_SL(Archive,len(z))
#     numberOfAdd = max(int(len(W)*add_ratio),1)
#     nad = min(len(SL),numberOfAdd)
#     for count in range(nad):
#         Y_sp = Y_[np.argmax(SL)]
#         Y_sp_ideal = sum([1/(Y_sp[i] -z[i]+epsilon) for i in range(len(z))])
#         X_add.append(X_[np.argmax(SL)])
#         Y_add.append(Y_sp)
#         W_add.append([(1/(Y_sp[i] - z[i]+epsilon))/Y_sp_ideal for i in range(len(z))])
#         SL.remove(max(SL))
#     X = np.append(X, np.array(X_add),axis=0)
#     Y = np.append(Y, np.array(Y_add),axis=0)
#     W = np.append(W, np.array(W_add),axis=0)
#     T = max(2, len(W)//10)
    
#     return X, Y, W, T

def add_vector(X, Y, W, z, Archive, add_ratio = 0.03,epsilon = 10**-7):
    W_add = []
    Y_add = []
    X_add = []
    Y_ = [item[0] for item in Archive]
    X_= [item[1] for item in Archive]
    SL = calc_SL(Archive,len(z))
    SL_list = [True for i in range(len(SL))]
    numberOfAdd = max(int(len(W)*add_ratio),1)
    nad = min(len(SL),numberOfAdd)
    for count in range(nad):
        Y_sp = Y_[np.argmax(np.array(SL)[SL_list])]
        Y_sp_ideal = sum([1/(Y_sp[i] -z[i]+epsilon) for i in range(len(z))])
        X_add.append(X_[np.argmax(SL)])
        Y_add.append(Y_sp)
        W_add.append([(1/(Y_sp[i] - z[i]+epsilon))/Y_sp_ideal for i in range(len(z))])
        SL_list[np.argmax(np.array(SL)[SL_list])] = False
    X = np.append(X, np.array(X_add),axis=0)
    Y = np.append(Y, np.array(Y_add),axis=0)
    W = np.append(W, np.array(W_add),axis=0)
    T = max(2, len(W)//10)
    
    return X, Y, W, T
    
def add_vector_random(X, Y, W, z, Archive, add_ratio = 0.03):
    W_add = []
    Y_add = []
    X_add = []
    numberOfAdd = max(int(len(W)*add_ratio),1)
    
    Y_ = [item[0] for item in Archive]
    X_= [item[1] for item in Archive]
    for count in range(numberOfAdd):
        rand_w = np.random.randn(len(z))
        rand_w = rand_w/sum(rand_w)
        tmp=[]
        for FV in Y_:
            tmp.append(Tchebychev(FV, rand_w, z))
        arg = np.argmin(tmp)
        
        W_add.append(rand_w)
        Y_add.append(Y_[arg])
        X_add.append(X_[arg])
        
    X = np.append(X, np.array(X_add),axis=0)
    Y = np.append(Y, np.array(Y_add),axis=0)
    W = np.append(W, np.array(W_add),axis=0)
    T = max(2, len(W)//10)
    
    return X,Y,W,T

def random_weights(X,Y,W,T, add_ratio=0.01):
    n_objs = len(W[0])
    numberOfAdd = max(int(len(W)*add_ratio),1)
    
    random_values = np.random.rand(numberOfAdd*50, n_objs)
    candidate = random_values/np.array([random_values.sum(axis=1) for i in range(n_objs)]).T
    
    dist = np.zeros(len(candidate))
    for i in range(len(candidate)):
        for item in W:
            dist[i]+= np.linalg.norm(item-candidate[i])
    arg_add = np.argsort(dist)[::-1][:numberOfAdd]
    
    W_add = []
    Y_add = []
    X_add = []
    
    for item in arg_add:
        tmp=[]
        for FV in Y_:
            tmp.append(Tchebychev(FV, candidate[item], z))
        arg = np.argmin(tmp)
        
        W_add.append(candidate[item])
        Y_add.append(Y_[arg])
        X_add.append(X_[arg])
        
    X = np.append(X, np.array(X_add),axis=0)
    Y = np.append(Y, np.array(Y_add),axis=0)
    W = np.append(W, np.array(W_add),axis=0)
    T = max(2, len(W)//10)
    
    return X,Y,W,T


def add_vector_fixed(X, Y, W, z, Archive, num_add,epsilon = 10**-7):
    if num_add == 0:
        T = max(2, len(W)//10)
        return X,Y,W,T
    W_add = []
    Y_add = []
    X_add = []
    Y_ = [item[0] for item in Archive]
    X_= [item[1] for item in Archive]
    SL = calc_SL(Archive,len(z))
    SL_list = [True for i in range(len(SL))]
    numberOfAdd = num_add
    nad = min(len(SL),numberOfAdd)
    for count in range(nad):
        Y_sp = Y_[np.argmax(np.array(SL)[SL_list])]
        Y_sp_ideal = sum([1/(Y_sp[i] -z[i]+epsilon) for i in range(len(z))])
        X_add.append(X_[np.argmax(SL)])
        Y_add.append(Y_sp)
        W_add.append([(1/(Y_sp[i] - z[i]+epsilon))/Y_sp_ideal for i in range(len(z))])
        SL_list[np.argmax(np.array(SL)[SL_list])] = False
    X = np.append(X, np.array(X_add),axis=0)
    Y = np.append(Y, np.array(Y_add),axis=0)
    W = np.append(W, np.array(W_add),axis=0)
    T = max(2, len(W)//10)
    
    return X, Y, W, T
    
def add_vector_random_fixed(X, Y, W, z, Archive, num_add):
    if num_add == 0:
        T = max(2, len(W)//10)
        return X,Y,W,T
    
    W_add = []
    Y_add = []
    X_add = []
    numberOfAdd = num_add
    
    Y_ = [item[0] for item in Archive]
    X_= [item[1] for item in Archive]
    for count in range(numberOfAdd):
        rand_w = np.random.randn(len(z))
        rand_w = rand_w/sum(rand_w)
        tmp=[]
        for FV in Y_:
            tmp.append(Tchebychev(FV, rand_w, z))
        arg = np.argmin(tmp)
        
        W_add.append(rand_w)
        Y_add.append(Y_[arg])
        X_add.append(X_[arg])
        
    X = np.append(X, np.array(X_add),axis=0)
    Y = np.append(Y, np.array(Y_add),axis=0)
    W = np.append(W, np.array(W_add),axis=0)
    T = max(2, len(W)//10)
    
    return X,Y,W,T
