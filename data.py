import numpy as np


def batch_normlize(x):
     """ This function is used to normalize an array/vector of data

    Parameters:
    x (array/vector): Array/vector of unnormalized numbers

    Returns:
    array/vector : normalized array/vector where normalized x = (x - mean) / np.sqrt(variance + eps)

   """
    eps=0.00001
    
    mean = x.mean(axis=0)
    variance = x.var(axis=0)
    
    xnormalized = (x - mean) / np.sqrt(variance + eps)
 
    return xnormalized


def flatten(x):
     """ This function is used to reshape a matrix into a flat shape.
     usefull in preproccessing of input data

    Parameters:
    x (matrix/vector): Multi-dimension matrix/vector

    Returns:
    matrix/vector: flatted matrix where each example is in a single row or column

   """
    return x.reshape(x.shape[0], -1).T

def onehot(y_train):
     """ This function is used to change data in the normal form into one-hot form 

    Parameters:
    y_train (matrix): Matrix of numbers

    Returns:
    matrix : One-hot form of the data

   """
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


