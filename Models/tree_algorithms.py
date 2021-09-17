import numpy as np
from base_model import BaseModel 

def find_gini(left, right, y):
    classes  = np.unique(y)
    n = len(left) + len(right)
    s1=0; s2=0
    
    for k in classes:   
        p1 = len(np.nonzero(y[left] == k)[0]) / len(left)
        s1 += p1*p1 
        p2 = len(np.nonzero(y[right] == k)[0]) / len(right)
        s2 += p2*p2 
    
    gini = (1-s1)*(len(left)/n) + (1-s2)*(len(right)/n)
    return gini

class DecisionTree(BaseModel):
    def __init__(self, x: np.ndarray, y: np.ndarray, idxs=None, min_leaf: int=5):  
        if idxs is None: 
            idxs=np.arange(len(y))

        self.x, self.y = x, y 
        self.idxs, self.min_leaf = idxs, min_leaf   
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()
        
    @property
    def split_name(self): return self.x.columns[self.var_idx]
    
    @property
    def split_cols(self): return self.x.values[self.idxs, self.var_idx]
    
    @property
    def is_leaf(self): return self.score == float('inf')
    
    def __repr__(self):
        s = f'n: {self.n}'
        if not self.is_leaf:
            s+= f'; gini:{self.score}; split:{self.split}; var: {self.split_name}'
        return s
            
    def check_features(self):
        
        for i in range(self.c): 
            self.find_best_split(i)
        if self.is_leaf: return 
        
        #otherwise this split becomes the root of a "new tree" 
        x = self.split_cols
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs]) 
        self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs])
    
    def find_best_split(self, var_idx):
    
        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]   
        sort_idx = np.argsort(x)
        sort_y = y[sort_idx]
        sort_x = x[sort_idx]

        for i in range(0, self.n-self.min_leaf-1):
            if i < self.min_leaf or sort_x[i] == sort_x[i+1]: continue 
            lhs = np.nonzero(sort_x <= sort_x[i])[0]
            rhs = np.nonzero(sort_x > sort_x[i])[0]
            if rhs.sum()==0: continue

            gini = find_gini(lhs, rhs, sort_y)

            if gini<self.score: 
                self.var_idx, self.score, self.split = var_idx, gini, sort_x[i]




class RandomForest():
    def __init__ (self, x: np.ndarray, y: np.ndarray, n_trees: int=5, sample_size=None, min_leaf: int =5):
        np.random.seed(42) 
        if sample_size is None:
            sample_size=len(y)

        self.x, self.y = x, y 
        self.sample_size, self.min_leaf = sample_size, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]
    
    def create_tree(self):
        idxs = np.random.choice(len(self.y), replace=True, size = self.sample_size)      
        return DecisionTree(self.x.iloc[idxs], 
                            self.y[idxs],
                            idxs=np.array(range(self.sample_size)),
                            min_leaf=self.min_leaf)
    
    def predict(self, x: np.ndarray):
        percents = np.mean([t.predict(x) for t in self.trees], axis=0)
        return [1 if p>0.5 else 0 for p in percents]