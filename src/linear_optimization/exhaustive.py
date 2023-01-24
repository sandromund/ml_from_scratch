import numpy as np
import itertools as it
from tqdm import tqdm


class EXHAUSTIVE:
    
    def __init__(self, L):
        
        self.L = L
        
        # STEP 1 
        self.A_ = np.append(L.A, np.identity(L.m) , axis=1) # AI_m € R^m x(n+m)
        self.c_ = np.append(L.c, np.zeros(L.m)) # c' = (c, 0) transpose?
        self.b = self.L.b

        self.x_best = None
        self.z_best = None
        self.B_best = None
        self.S = self.__B_set()
        self.A_B_inv = None

        self.__solved = False
        self.__zulaessig = None
        print(self.info())
        
    def __B_set(self):       
        S = set(range(self.L.m + self.L.n)) # [n + m] = {0, ... n+m-1}
        S = set(it.combinations(S, self.L.m)) # S € [n + m] mit #B = m
        S = {B for B in S if np.linalg.det(self.A_[:, B]) != 0} # check if regulär
        return S
    
    def run(self):
        print("\n=== Starte EXHAUSTIVE suche ===")
        k = self.L.n + self.L.m
        for B in tqdm(self.S):
            A_B_inv = np.linalg.inv(self.A_[:, B])
            x_B = A_B_inv  @ self.b
            x_ = np.zeros(k)
            for index, value in zip(B, x_B):
                x_[index] = value[0]
            x_ = x_.reshape(k, 1)
            
            z = np.matmul(self.c_, x_).round(10)[0]
            c = (self.x_best is None or self.z_best is None) 
            if all(x_ >= 0):
                #print("                   B:", B , " z:", z)
                if (c or z > self.z_best):
                    print("Besseres B:", B, "mit z=", z, "gefunden. ")
                    self.x_best = x_
                    self.z_best = z
                    self.B_best = B
                    self.A_B_inv = A_B_inv
                    self.x_B = x_B

        if (self.x_best is None or self.z_best is None):
            print("UNZULÄSSIG!")
        else:
            self.__zulaessig = True
        
        self.__solved = True
        print("=== Ende EXHAUSTIVE suche ===\n")
        self.result()

    
    def info(self):
        s =  "EXHAUSTIVE: %s \n"
        s += "A': \n %s \n"
        s += "b:  \n %s \n"
        s += "c': \n %s "
        return s % (self.L.name, self.A_, self.L.b, self.c_)
        
    
    def result(self):
        assert self.__solved is True
        
        if self.__zulaessig is None:
            return 
               
        print("Basis B: \n", self.B_best)
        print("Basislösung x': \n", self.x_best)
        print("z = c'x': \n ", self.z_best)
        print("A_B: \n", self.A_[:, self.B_best])
        print("A_B_inv:\n", self.A_B_inv )
        print("X_B:\n", self.x_B)

              
    def __str__(self):
        return self.info()
    
    def __repr__(self):
        return self.__str__()
