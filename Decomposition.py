import numpy as np

def Tchebychev(F,Vector,Ideal,Constraint=[],min_weight=0.001):
    return max([max(Vector[i],min_weight)*(F[i]-Ideal[i]) for i in range(len(F))])

def pbi(f,Vector,Ideal,Constraint=[],theta=5):
    w = np.array(Vector)
    z_star = np.array(Ideal)
    F = np.array(f)

    d1 = np.linalg.norm(np.dot((F - z_star), w)) / np.linalg.norm(w)
    d2 = np.linalg.norm(F - (z_star + d1 * w))

    return (d1 + theta * d2).tolist()