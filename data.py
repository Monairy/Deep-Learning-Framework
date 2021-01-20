import numpy as np


def batch_normlize(x):
    eps=0.00001
    
    mean = x.mean(axis=0)
    variance = x.var(axis=0)
    
    xnormalized = (x - mean) / np.sqrt(variance + eps)
 
    return xnormalized

X = np.array(
           [
                      [1,1],
                      [2,2],
                      ]
           )

y=batch_normlize(X)
print(y)
