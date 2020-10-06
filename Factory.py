from pymop.factory import get_problem
import numpy as np

class Problem:
    def __init__(self, prob_name, n_vars, n_objs):
        self.prob_name = prob_name
        self.n_vars = n_vars
        self.n_objs = n_objs
        
    def do(self, X):
        prob = get_problem(self.prob_name, n_var=self.n_vars, n_obj = self.n_objs)
        return prob.evaluate(X)
#         return prob.evaluate(X),[]         # if Constraint
    
    