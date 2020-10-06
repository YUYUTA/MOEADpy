import numpy as np
from sklearn.neighbors import NearestNeighbors

def calc_SL(W,n_objs=3):
    num = min(n_objs,len(W))

    neigh = NearestNeighbors(num)
    neigh.fit(W)

    neighbors = neigh.kneighbors(W, num)[0].tolist()
    SL = [np.prod(item[1:num]) for item in neighbors]
    return SL

def myParetoDominance(solution1,solution2):
    dominate1 = False
    dominate2 = False
    
    for i in range(len(solution1[0])):
        o1 = solution1[0][i]
        o2 = solution2[0][i]
        if o1 < o2:
            dominate1 = True

            if dominate2:
                return 0
        elif o1 > o2:
            dominate2 = True

            if dominate1:
                return 0
            
    if dominate1 == dominate2:
        return 0
    elif dominate1:
        return -1
    else:
        return 1
    
# def update_archive(archive,solution):
#     add_flag = False
#     remove_list = []
#     L = len(archive)
#     for i in range(L):
#         flag = myParetoDominance(solution,archive[i])
#         if flag == 1:
#             return archive
#         elif flag == -1:
#             remove_list.append(i)
#             add_flag = True
#         elif flag == 0:
#             add_flag = True

#     if len(archive)==0:
#         add_flag = True
    
#     if add_flag:
#         archive.append(solution)
#     if remove_list:
#         archive = np.delete(np.array(archive),remove_list,axis=0).tolist()

def update_archive(archive,solution):
    add_flag = False
    remove_list = []
    L = len(archive)
    for i in range(L):
        flag = myParetoDominance(solution,archive[i])
        if flag == 1:
            return archive
        elif flag == -1  or sum(np.array(solution[0]) == np.array(archive[i][0]))==len(solution[0]):
            remove_list.append(i)
            add_flag = True
        elif flag == 0:
            add_flag = True

    if len(archive)==0:
        add_flag = True
    
    if add_flag:
        archive.append(solution)
    if remove_list:
        archive = np.delete(np.array(archive),remove_list,axis=0).tolist()
        
    return archive


def init_archive(pop, archive = []):
    for item in pop:
        archive = update_archive(archive, item)
        
    return archive

def update_AWA_archive(archive, pop_size):
    archive_size = 2*pop_size
    objs = np.array([item[0] for item in archive])
    N = len(objs) - archive_size
    if len(objs) <= archive_size:
        return archive
    remove_list = []
    while len(objs)> archive_size:
        objs = np.array([item[0] for item in archive])
        SL = calc_SL(objs)
        arg = np.argmin(SL)
        index = np.ones(len(objs), dtype=bool)
        index[arg] = False
        archive = np.array(archive)[index].tolist()
    
    return np.delete(np.array(archive),remove_list,axis=0).tolist()
    
    
    