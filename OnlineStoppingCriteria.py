import numpy as np

class CR:
    def __init__(self, delta=10, F=10):
        self.archive = []
        self.CR_archive = []
        self.Ui_archive = []
        self.Uinit = -1
        self.gen = 0
        self.delta = delta
        self.F = F
        
    def paretoDominance(self, solution1,solution2):
        dominate1 = False
        dominate2 = False

        for i in range(len(solution1)):
            o1 = solution1[i]
            o2 = solution2[i]
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
        
    def calc_CR(self):
        n_Ai = len(self.archive[-1])
        n_S = 0
        for item1 in self.archive[0]:
            for item2 in self.archive[-1]:
                if self.paretoDominance(item1,item2)==1:
                    break
            else:
                n_S+=1
                
        return n_S/n_Ai

    def calc_Ui(self):
        if self.CR_archive[-1] <= 0.5:
            return 10**9
        return (self.CR_archive[-1] - self.CR_archive[0])/self.delta

    def Ui_star(self):
        return (self.Ui_archive[-1]+self.Ui_archive[0])/2

    def Ut(self):
        if self.CR_archive[-1]> 0.5 and self.Uinit==-1:
            self.Uinit = self.CR_archive[-1]/self.gen
        return self.Uinit/self.F

    def update_archive(self, Y):
        self.gen+=1
        if self.gen < self.delta:
            self.archive.append(Y)
            self.CR_archive.append(self.calc_CR())
            self.Ui_archive.append(self.calc_Ui())
        else:
            self.archive.append(Y)
            self.archive.pop(0)
            
            self.CR_archive.append(self.calc_CR())
            self.CR_archive.pop(0)
            
            self.Ui_archive.append(self.calc_Ui())
            self.Ui_archive.pop(0)
    
    def OnlineStoppingCriteria(self, Y):
        self.update_archive(Y)
#         flag = self.CR_archive[-1]>0.9 #0.8
        flag = self.Ui_star() < self.Ut()
        if flag:
            self.archive = []
            self.CR_archive = []
            self.Ui_archive = []
            self.Uinit = -1
            self.gen = 0
            return True
        return False
    
        

