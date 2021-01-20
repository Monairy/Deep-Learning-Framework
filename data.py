import numpy as np


def batch_normlize(x):
    eps=0.00001
    
    mean = x.mean(axis=0)
    variance = x.var(axis=0)
    
    xnormalized = (x - mean) / np.sqrt(variance + eps)
 
    return xnormalized


def flatten(x):
    return x.reshape(x.shape[0], -1).T

def onehot(y_train)

  y = np.zeros((y_train.size, y_train.max()+1))
  y  [np.arange(y_train.size),y_train] = 1
  y = y.T
  return y


X = np.array(
           [
                      [1,1],
                      [2,2],
                      ]
           )

y=batch_normlize(X)
print(y)


