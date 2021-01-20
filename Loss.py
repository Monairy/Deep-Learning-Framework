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
    #if Y * A > 0:
    #    return 0
    #else:
    #    return - np.gradient(A)
    p = Y * A
    b = np.zeros(A.shape)
    b[p > 0] = 0
    b[p <= 0] = -Y[p <= 0]
    b[b == 0] = -1
    return b

def svm(m, A, Y):
    cost = (1 / m) * np.sum(np.maximum(0, 1 - Y * A))
    return cost

def svm_der(m, A, Y):
    #if Y * A - 1 > 0:
    #    return 0
    #else:
    #    return - np.gradient(A)
    p = Y * A - 1
    b = np.zeros(A.shape)
    b[p > 0] = 0
    b[p <= 0] = -Y[p <= 0]
    b[b == 0] = -1
    return b

def cross_multi_class(m, A, Y): # Multiclass Log LikelihoodLoss Function - Logistic Regression SoftMax Activation Function
    # v1 = Y * A
    # v2 = np.max(v1,axis=0)
    # v3 = np.log(v2).sum()
    # return (-1 / m) * v3

    cost = (-1 / m) * np.sum((Y) * (np.log(A)))
    # print("in cross multi")
    return cost


def cross_multi_class_der(m, A, Y):
    z1 = np.array(A, copy=True)
    y1 = np.array(Y, copy=True)
    y1[y1 == 1] = -1
    return A - Y

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

def multinomial_logistic_loss(m, A, Y):
    cost = np.sum(-np.max(A * Y) + np.log(np.sum(np.exp(A))))
    return cost


def multinomial_logistic_loss_der(m, A, Y):
    p = np.zeros(A.shape)
    p[Y == 1] = -(1 - A[Y == 1])
    p[Y == 0] = A[Y == 0]
    return p

def square_loss(m, A, Y):
    cost = (1/2*m) * np.sum(np.square(Y - A))
    return cost


def square_loss_der(m, A, Y):
    return (-1/m) * (Y - A)


def logistic_sigmoid_loss(m, A, Y):
    cost = (-1/m) * np.sum(np.log(0.5*Y - 0.5 + A))
    return cost


def logistic_sigmoid_loss_der(m, A, Y):
    return (- 1) / (0.5*Y - 0.5 + A)


def logistic_id_loss(m, A, Y):
    cost = (1 / m) * np.sum(np.log(1 + np.exp(- Y * A)))
    return cost


def logistic_id_loss_der(m, A, Y):
    return (1 / m) * (- Y * np.exp(- Y * A)) / (1 + np.exp(- Y * A))


def determine_der_cost_func(func):
    if func == cross_entropy:
        return cross_entropy_der
    if func == cross_multi_class:
        return cross_multi_class_der
    if func == square_loss:
        return square_loss_der
    if func == perceptron_criteria:
        return perceptron_criteria_der
    if func == svm:
        return svm_der
    if func == multiclass_perceptron_loss:
        return multiclass_perceptron_loss_der
    if func == multiclass_svm:
        return multiclass_svm_der
    if func == multinomial_logistic_loss:
        return multinomial_logistic_loss_der
