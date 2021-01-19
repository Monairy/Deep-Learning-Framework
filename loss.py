import numpy as np
# m -> no of examples
# A -> Output layer (y_hat)
# Y -> Label

def cross_entropy(m, A, Y): # Log LikelihoodLoss Function - Logistic Regression Sigmoid Activation Function
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    return cost

def cross_entropy_der(m, A, Y):
    return ((-1 * Y) / A) + ((1 - Y) / (1 - A))

def perceptron_criteria(m, A, Y):
    cost = (1 / m) * np.sum(np.maximum(0, - Y * A))
    return cost

def perceptron_criteria_der(m, A, Y):
    #return np.maximum()
    if Y * A > 0:
        return 0
    else:
        return - np.gradient(A)

def svm(m, A, Y):
    cost = (1 / m) * np.sum(np.maximum(0, 1 - Y * A))
    return cost

def svm_der(m, A, Y):
    if Y * A - 1 > 0:
        return 0
    else:
        return - np.gradient(A)

def cross_multi_class(m, A, Y): # Multiclass Log LikelihoodLoss Function - Logistic Regression SoftMax Activation Function
    # v1 = Y * A
    # v2 = np.max(v1,axis=0)
    # v3 = np.log(v2).sum()
    # return (-1 / m) * v3

    cost = (-1 / m) * np.sum((Y) * (np.log(A)))
    # print("in cross multi")
    return cost

def multiclass_perceptron_loss(m, A, Y):
    D = np.maximum(A - np.max(Y*A), 0)
    cost = (1 / m) * np.sum(np.max(D))
    return cost

def multiclass_perceptron_loss_der(m, A, Y):
    if np.arange(np.shape(A)) == np.argmax(Y*A):
        return - np.gradient(np.max(Y*A))
    elif np.arange(np.shape(A)) != np.argmax(Y*A):
        return np.gradient(A)
    else:
        return 0

def multiclass_svm(m, A, Y):
    D = np.maximum(1 + A - np.max(Y*A), 0)
    cost = (1 / m) * np.sum(np.sum(D))
    return cost

def multiclass_svm_der(m, A, Y):
    if np.arange(np.shape(A)) == np.argmax(Y*A):
        return - np.gradient(np.max(Y*A))
    elif np.arange(np.shape(A)) != np.argmax(Y*A):
        return np.gradient(A)
    else:
        return 0

# def multinomial_logistic_regression_loss(m, A, Y):
#    cost = (-1 / m) * np.sum((Y) * (np.log(A)))

def cross_multi_class_der(m, A, Y):
    z1 = np.array(A, copy=True)
    y1 = np.array(Y, copy=True)
    y1[y1 == 1] = -1
    return A - Y


def determine_der_cost_func(func):
    if func == cross_entropy:
        return cross_entropy_der
    if func == cross_multi_class:
        return cross_multi_class_der
