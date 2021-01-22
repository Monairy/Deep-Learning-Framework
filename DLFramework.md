Module data
===========

Functions
---------

    
`addpoint_y(self, pointy)`
:   This function is used to add a new point to be drawn on a graph 
    
    Parameters:
      
       pointy (float): value of y coordinate 
    
    Returns:
      
       graph: graph drawn after adding new point

    
`batch_normlize(x)`
:   This function is used to normalize an array/vector of data
    
    Parameters:
      
       x (array/vector): Array/vector of unnormalized numbers
    
    Returns:
      
       (array/vector) : normalized array/vector where normalized x = (x - mean) / np.sqrt(variance + eps)

    
`flatten(x)`
:   This function is used to reshape a matrix into a flat shape.
      usefull in preproccessing of input data
    
    Parameters:
      
       x (matrix/vector): Multi-dimension matrix/vector
    
    Returns:
     
       (matrix/vector): flatted matrix where each example is in a single row or column

    
`onehot(y_train)`
:   This function is used to change data in the normal form into one-hot form 
    
    Parameters:
     
       y_train (matrix): Matrix of numbers
    
    Returns:
      
       (matrix) : One-hot form of the data

    
`retrive(filename)`
:   This function loads a saved model ( all of it's parameters, weights, losses, ...) from external file
    
    Parameters:
       
       filename (string): Path of file to be retrived, filename must be ".sav" type.
    
    Returns:
       
       (object): Object of model that saved before as whole with its structure(Layers,Nodes,Activation functions)

    
`save(filename, model)`
:   This function saves the model ( all of it's parameters, weights, losses ,... ) in an external file 
    
    Parameters:
       
       filename (string): Path of file to be saved in, filename must be ".sav" type.
       model    (object): "model" required to be saved.
    
    Returns:
       (None)

Classes
-------

`visualize()`
:   

    ### Class variables

    `fig`
    :

    `graph`
    :

    `points_x`
    :

    `points_y`
    :Module frame
============

Classes
-------

`MultiLayer(number_of_neurons=0, cost_func=<function cross_entropy>)`
:   init of the class multilayer and needed variables
    variables:
        w,b lists for weights
        parameters dic for weights in the form of parameters['W1']
        layers_size for size of each layer
        number_of_input_neurons
        act_func list for activations of each layer
        derivative_act_func list for backward activations derivative functions
        cost_func the choosen cost functions
    
    parmeters:
        (method) : the cost function of model
    
    returns:
        (None)

    ### Methods

    `addHidenLayer(self, size, act_func=<function sigmoid>)`
    :   add a hidden layer of the model
        
        parmeters:
            size (int) : size of input layer
            act_func (function) : the activation function of the layer
        
        retruns:
            (None)

    `addLayerInput(self, size)`
    :   add the input layer of the model
        
        parmeters:
            size (int) : size of input layer
        
        retruns:
            (None)

    `addOutputLayer(self, size, act_func=<function sigmoid>)`
    :   add the output layer of the model
        
        parmeters:
            size (int) : size of input layer
            act_func (function) : the activation function of the layer
        
        retruns:
            (None)

    `backward_propagation(self, X, Y)`
    :   compute cost of the given examples
        
        parmeters:
            Alast (np.array) : model predictions
            Y (np.array) : True labels
        
        retruns:
            grads (dic) : all gridients of wieghts and biasses

    `compute_cost(self, Alast, Y)`
    :   compute cost of the given examples
        
        parmeters:
            Alast (np.array) : model predictions
            Y (np.array) : True labels
        
        retruns:
            cost (float) : cost output

    `forward_propagation(self, X, drop=0)`
    :   forward propagation through the layers
        
        parmeters:
            X (np.array) : input feature vector
            drop (float) : propablity to keep neurons or shut down
        
        retruns:
            cashe (dic) : the output of each layer in the form of cashe['Z1']
            Alast (np.array) : last layer activations

    `initialize_parameters(self, seed=2)`
    :   initialize_weights of the model at the start with xavier init
        
        parmeters:
            seed (int) : seed for random function
        
        retruns:
            paramters

    `predict(self, X)`
    :   perdict classes or output
        
        parmeters:
            X (np.array) :  input feature vector
        
        retruns:
            Alast (np.array) : output of last layer

    `set_cashe(self, cache, X)`
    :   set an external cache
        
        parmeters:
            X (np.array) : input feature vector
            cache (dic) :  output of each layer
        
        retruns:
            (None)

    `set_cost(self, cost_func)`
    :   cahnge the initial cost function
        
        parmeters:
            cost_funct (function) : the new function
        
        retruns:
            cashe (dic) : the output of each layer in the form of cashe['Z1']
            Alast (np.array) : last layer activations

    `set_parameters(self, para)`
    :   set an external parmeters
        
        parmeters:
            para (dic) :  the weights and biasses
        
        retruns:
            (None)

    `set_parameters_internal(self)`
    :   set an internal parmeters this is used by model during training
        
        parmeters:
            (None)
        
        retruns:
            (None)

    `test(self, X, Y, eval_func=<function accuracy_score>)`
    :   evalute model
        
        parmeters:
            X (np.array) :  input feature vector
            Y (np.array) :  the true label
            eval_func (function) : the method of evalution
        
        retruns:
            Alast (np.array) : output of last layer

    `train(self, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1, reg_term=0, batch_size=0, opt_func=<function gd_optm>, param_dic=None, drop=0)`
    :   train giving the data and hpyerparmeters and optmizer type
        
        parmeters:
            X (np.array) : input feature vector
            Y (np.array) :  the true label
            num_of iterations (int) : how many epochs
            print cost (bool) : to print cost or not
            print cost_each (int) : to print cost each how many iterations
            learning_rate (float) : the learn rate hyper parmeter
            reg_term (float) : the learn rate hyper parmeter
            batch_size (int) : how big is the mini batch and 0 for batch gradint
            optm_func (function) : a function for calling the wanted optmizer
        
        retruns:
            parmeters (dic) : weights and biasses after training
            cost (float) : cost

    `upadte_patameters_RMS(self, grads, rmsgrads, learning_rate=1.2, reg_term=0, m=1, eps=None)`
    :   update parameters using RMS gradient
        
        parameters:
            grads (dic) :  the gradient of weights and biases
            rmsgrads(dic): taking rho multiplied by the square of previous grads and (1-rho) multiplied by the square of current grads
            learning_rate (float) : the learn rate hyper parameter
            reg_term (float) : the learn rate hyper parameter
            eps(float) : the small value added to rmsgrads to make sure there is no division by zero
        
        returns:
            dictionary contains the updated parameters

    `upadte_patameters_adadelta(self, grads, delta, learning_rate=1.2, reg_term=0, m=1)`
    :   update parameters using RMS gradient
        
        parameters:
            grads (dic) :  the gradient of weights and biases, note: this parameter is not used in this function
            delta(dic): dictionary contains the values that should be subtracted from current parameters to be updated
            learning_rate (float) : the learn rate hyper parameter , note: this parameter is not used in this function
            reg_term (float) : the learn rate hyper parameter
        
        returns:
            dictionary contains the updated parameters

    `update_parameters(self, grads, learning_rate=1.2, reg_term=0, m=1)`
    :   update parameters using grads
        
        parmeters:
            grads (dic) :  the gradient of weights and biases
            learning_rate (float) : the learn rate hyper parameter
            reg_term (float) : the learn rate hyper parameter
        
        returns:
            dictionary contains the updated parameters

    `update_parameters_adagrad(self, grads, adagrads, learning_rate=1.2, reg_term=0, m=1)`
    :   update parameters using adagrad
        
        parameters:
            grads (dic) :  the gradient of weights and biases
            adagrads(dic): the square of the gradiant
            learning_rate (float) : the learn rate hyper parameter
            reg_term (float) : the learn rate hyper parameter
        
        returns:
            dictionary contains the updated parameters

    `update_parameters_adam(self, grads, adamgrads, Fgrads, learning_rate=1.2, reg_term=0, m=1, eps=None)`
    :   update parameters using RMS gradient
        
        parameters:
            grads (dic) :  the gradient of weights and biases , note: grads is not used in this function
            adamgrads(dic): taking rho multiplied by the square of previous grads and (1-rho) multiplied by the square of current grads
            Fgrads(dic): taking rhof multiplied by the  previous grads and (1-rhof) multiplied by the  current grads
            learning_rate (float) : the learn rate hyper parameter (alpha_t not alpha)
            reg_term (float) : the learn rate hyper parameter
            eps(float) : the small value added to adamgrads to make sure there is no division by zero
        
        returns:
            dictionary contains the updated parametersModule activations
==================

Functions
---------

    
`determine_der_act_func(func)`
:   This function works as a switch, returns the right dervative function for backpropagation as an opposite operation of applied activation function in forwardpropagation.
    
    Parameters:
        
        func (method): The activation function used in forwardpropagation.
    
    Returns:
        
        (method): Returning method of selective derivative activation function to make backpropagation

    
`identity(z)`
:   This function applies identity function(has no mathematical form) as an activation function of a node for forwardpropagation.
    
    Parameters:
        
        z (numpy array): result of W.X (Weights.Inputs/features).
    
    Returns:
        
        (numpy array): Returning same array of input "z".

    
`identity_der(z)`
:   This function applies identity derivative function(has no mathematical form) for backpropagation.
    
    Parameters:
        
        z (numpy array): result of W.X (Weights.Inputs/features).
    
    Returns:
        
        (int): Returning 1 as its derivative of z.

    
`relu(z)`
:   This function applies relu function(has mathematical form) as an activation function of a node for forwardpropagation.
    
    Parameters:
        
        z (numpy array): result of W.X (Weights.Inputs/features).
    
    Returns:
        
        (numpy array): Returning array of same size as input "z" after applying "max(0,input)".

    
`relu_der(A)`
:   This function applies relu derivative function(has mathematical form = 1) for backpropagation
    
    Parameters:
        
        A (numpy array): result of W.X (Weights.Inputs/features)
    
    Returns:
        
        (numpy array): Returning array of same size as input "A", has 2 conditions; if less than zero then 0, else 1.

    
`sigmoid(z)`
:   This function applies sigmoid function(has mathematical form) as an activation function of a node for forwardpropagation.
    
    Parameters:
        
        z (numpy array): result of W.X (Weights.Inputs/features).
    
    Returns:
        
        (numpy array): Returning array of same size as input "z" after applying sigmoid on input.

    
`sigmoid_der(A)`
:   This function applies sigmoid derivative function(has mathematical form) for backpropagation.
    
    Parameters:
        
        A (numpy array): result of W.X (Weights.Inputs/features).
    
    Returns:
        
        (numpy array): Returning array of same size as input "A".

    
`softmax(z)`
:   This function applies softmax function(has mathematical form) as an activation function of a node for forwardpropagation.
    
    Parameters:
        
        z (numpy array): result of W.X (Weights.Inputs/features).
    
    Returns:
        
        (numpy array): Returning array of same size as input "z" after applying "exp()/sum of exponential of all inputs".

    
`softmax_der(A)`
:   This function applies softmax derivative function(has mathematical form = 1) for backpropagation.
    
    Parameters:
        
        A (numpy array): result of W.X (Weights.Inputs/features).
    
    Returns:
        
        (int): Returning 1.

    
`tanh(z)`
:   This function applies tanh function(has mathematical form) as an activation function of a node for forwardpropagation.
    
    Parameters:
        
        z (numpy array): result of W.X (Weights.Inputs/features).
    
    Returns:
        
        (numpy array): Returning array of same size as input "z" after applying "tanh()".

    
`tanh_der(A)`
:   This function applies tanh derivative function(has mathematical form) for backpropagation.
    
    Parameters:
        
        A (numpy array): result of W.X (Weights.Inputs/features).
    
    Returns:
        
        (numpy array): Returning array of same size as input "A" after applying derivative of tanh().Module optmizers
================

Functions
---------

    
`RMS_optm(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1, reg_term=0, batch_size=0, param_dic=None, drop=0)`
:   The function applies the RMS optimizer to update the weight and bias parameters.
    
    Parameters:
       
    model (multilayer): instance of the multilayer class contains the models parameters to be updated.
    X: the input feature vector.
    Y: the labels.
    num_iterations: number of epochs.
    print_cost: optional parameter to show the cost function.
    print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
    cont: not used in this function
    learning_rate: learning rate to be used in updating the parameters.
    reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
    batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
    param_dic: the dictionary that contains the value of the hyper parameter rho
    drop: dropout parameter to have the option of using the dropout technique.
    
    Returns:
        
    dictionary:parameters a dictionary that contains the updated weights and biases
    array:Costs an array that contain the cost of each iteration

    
`adadelta_optm(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1, reg_term=0, batch_size=0, param_dic=None, drop=0)`
:   The function applies the adadelta optimizer to update the weight and bias parameters.
    
    Parameters:
        
        model (multilayer): instance of the multilayer class contains the models parameters to be updated.
        X: the input feature vector.
        Y: the labels.
        num_iterations: number of epochs.
        print_cost: optional parameter to show the cost function.
        print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
        cont: not used in this function
        learning_rate: learning rate to be used in updating the parameters.
        reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
        batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
        param_dic: the dictionary that contains the value of the hyper parameters rho and epsilon.
        drop: dropout parameter to have the option of using the dropout technique.
    
    Returns:
    
        dictionary: parameters a dictionary that contains the updated weights and biases
        array: Costs an array that contain the cost of each iteration

    
`adagrad_optm(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1, reg_term=0, batch_size=0, param_dic=None, drop=0)`
:   The function applies the adagrad optimizer to update the weight and bias parameters.
    
    Parameters:
       
        model (multilayer): instance of the multilayer class contains the models parameters to be updated.
        X: the input feature vector.
        Y: the labels.
        num_iterations: number of epochs.
        print_cost: optional parameter to show the cost function.
        print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
        cont: not used in this function
        learning_rate: learning rate to be used in updating the parameters.
        reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
        batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
        param_dic: not used in this function.
        drop: dropout parameter to have the option of using the dropout technique.
    
    Returns:
       
        dictionary: parameters a dictionary that contains the updated weights and biases
        array: Costs an array that contain the cost of each iteration

    
`adam_optm(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1, reg_term=0, batch_size=0, param_dic=None, drop=0)`
:   The function applies the adam optimizer to update the weight and bias parameters.
    
    Parameters:
        
        model (multilayer): instance of the multilayer class contains the models parameters to be updated.
        X: the input feature vector.
        Y: the labels.
        num_iterations: number of epochs.
        print_cost: optional parameter to show the cost function.
        print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
        cont: not used in this function
        learning_rate: learning rate to be used in updating the parameters.
        reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
        batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
        param_dic: the dictionary that contains the value of the hyper parameters rho , rhof and epsilon.
        drop: dropout parameter to have the option of using the dropout technique.
    
    Returns:
        
        dictionary:parameters a dictionary that contains the updated weights and biases
        array:Costs an array that contain the cost of each iteration

    
`gd_optm(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1, reg_term=0, batch_size=0, param_dic=None, drop=0)`
:   The function applies the  gradient descent optimizer to update the weight and bias parameters.
    
    Parameters:
       
        model (multilayer): instance of the multilayer class contains the models parameters to be updated.
        X: the input feature vector.
        Y: the labels.
        num_iterations: number of epochs.
        print_cost: optional parameter to show the cost function.
        print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
        cont: not used in this function
        learning_rate: learning rate to be used in updating the parameters.
        reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
        batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
        param_dic: not used in this function.
        drop: dropout parameter to have the option of using the dropout technique.
    
    Returns:
        
        dictionary:parameters a dictionary that contains the updated weights and biases
        array:Costs an array that contain the cost of each iteration

    
`gd_optm_steepst(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=0.01, reg_term=0, batch_size=0, param_dic=None, drop=0)`
:   The function applies the steepest gradient descent optimizer to update the weight and bias parameters.
               
    Parameters:
                
        model (multilayer): instance of the multilayer class contains the models parameters to be updated.
        X: the input feature vector.
        Y: the labels.
        num_iterations: number of epochs.
        print_cost: optional parameter to show the cost function.
        print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
        cont: not used in this function
        learning_rate: learning rate to be used in updating the parameters.
        reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
        batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
        param_dic: not used in this function.
        drop: dropout parameter to have the option of using the dropout technique.
               
    Returns:
        
        dictionary:parameters a dictionary that contains the updated weights and biases
        array:Costs an array that contain the cost of each iteration

    
`mom_optm(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1, reg_term=0, batch_size=0, param_dic=None, drop=0)`
:   The function applies the momentum optimizer to update the weight and bias parameters.
    
    Parameters:
                
        model (multilayer): instance of the multilayer class contains the models parameters to be updated.
        X: the input feature vector.
        Y: the labels.
        num_iterations: number of epochs.
        print_cost: optional parameter to show the cost function.
        print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
        cont: not used in this function
        learning_rate: learning rate to be used in updating the parameters.
        reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
        batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
        param_dic: the dictionary that contains the value of the hyper parameters beta.
        drop: dropout parameter to have the option of using the dropout technique.
    
    Returns:
        
        dictionary:parameters a dictionary that contains the updated weights and biases
        array:Costs an array that contain the cost of each iterationModule Loss
===========

Functions
---------

    
`cross_entropy(m, A, Y)`
:   Log LikelihoodLoss Function - Logistic Regression Sigmoid Activation Function
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns: 
        cost(float): the total loss

    
`cross_entropy_der(m, A, Y)`
:   The Derivative of Log LikelihoodLoss Function
    
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns:
        (Array of floats): The derivative values of cost function

    
`cross_multi_class(m, A, Y)`
:   Multiclass Log LikelihoodLoss Function - Logistic Regression SoftMax Activation Function
    
    Parameters:
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns: 
        cost(float): the total loss

    
`cross_multi_class_der(m, A, Y)`
:   The Derivative of Multiclass Log LikelihoodLoss Function
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns:
        (Array of floats): The derivative values of cost function

    
`determine_der_cost_func(func)`
:   Determining The Derivative of The Loss function
    
    Parameters: 
        func(string): The Loss function name
    
    Returns:
        (string): The Derivative of The Loss function

    
`logistic_id_loss(m, A, Y)`
:   Logistic Regression using Identity Activation
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns: 
        cost(float): the total loss

    
`logistic_id_loss_der(m, A, Y)`
:   The Derivative of Logistic Regression using Identity Activation Loss Function
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns:
        (Array of floats): The derivative values of cost function

    
`logistic_sigmoid_loss(m, A, Y)`
:   Logistic Regression using sigmoid Activation
    
    Parameters: 
        m(int):examples no.
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns:
        cost(float): the total loss

    
`logistic_sigmoid_loss_der(m, A, Y)`
:   The Derivative of Logistic Regression using sigmoid Activation Loss Function
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns:
        (Array of floats): The derivative values of cost function

    
`multiclass_perceptron_loss(m, A, Y)`
:   Multiclass Perceptron Loss
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns: 
        cost(float): the total loss

    
`multiclass_perceptron_loss_der(m, A, Y)`
:   The Derivative of Multiclass Perceptron Loss Function
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)   
        Y(float vector): The label
    
    Returns:
        p(Array of floats): The derivative values of cost function

    
`multiclass_svm(m, A, Y)`
:   Multiclass Weston-Watkins SVM Loss
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns: 
        cost(float): the total loss

    
`multiclass_svm_der(m, A, Y)`
:   The Derivative of Multiclass Weston-Watkins SVM Loss Function
    
    Parameters: 
        m (int):examples no. 
        A (float vector): The output y_hat (score)  
        Y (float vector): The label
    
    Returns: 
        p (Array of floats): The derivative values of cost function

    
`multinomial_logistic_loss(m, A, Y)`
:   Multinomial Logistic Regression using Softmax Activation
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns: 
        cost(float): the total loss

    
`multinomial_logistic_loss_der(m, A, Y)`
:   The Derivative of Multinomial Logistic Regression using Softmax Activation Loss Function
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns:
        (Array of floats): The derivative values of cost function

    
`perceptron_criteria(m, A, Y)`
:   Perceptron Criteria
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns:
        cost(float): the total loss

    
`perceptron_criteria_der(m, A, Y)`
:   The Derivative of Perceptron Criteria loss Function
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns:
        b(Array of floats): The derivative values of cost function

    
`square_loss(m, A, Y)`
:   Linear Regression Least squares using Identity Activation
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns: 
        cost(float): the total loss

    
`square_loss_der(m, A, Y)`
:   The Derivative of Linear Regression Least squares using Identity Activation Loss Function
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns:
    (Array of floats): The derivative values of cost function

    
`svm(m, A, Y)`
:   Hinge Loss (Soft Margin) SVM
    
    Parameters:
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns: 
        cost(float): the total loss

    
`svm_der(m, A, Y)`
:   The Derivative of Hinge Loss (Soft Margin) SVM Function
    
    Parameters: 
        m(int):examples no. 
        A(float vector): The output y_hat (score)  
        Y(float vector): The label
    
    Returns:
        b(Array of floats): The derivative values of cost functionModule evaluation
=================

Functions
---------

    
`accuracy_score(A, Y, thres=0.5)`
:   This function calculates the accuracy of the model using predicted and truth values
    
    Parameters:
        A (numpy matrix): predicted classification for each example
        Y (numpy matrix): true classification for each example
        thres (float): threshold value, values above it are considered true
    
    Returns:
        (float): accuracy

    
`conf_mat(A, Y, thres=0.5)`
:   This function calculates the confusion matrix from predicted and truth matrices
    
    Parameters:
        A (numpy matrix): predicted classification for each example
        Y (numpy matrix): true classification for each example
        thres (float): threshold value, values above it are considered true
    
    Returns:
        (numpy matrix): confusion matrix

    
`conf_table(cnf_matrix)`
:   Tis function calculates the confusion table from confusion matrix
    
    Parameters:
        cnf_matrix (numpy matrix): confusion matrix
    
    Returns:
        (tuple of floats): representing TP for each class
        (tuple of floats): representing FP for each class
        (tuple of floats): representing TN for each class
        (tuple of floats): representing FN for each class

    
`f1_score(A, Y, thres=0.5)`
:   This function calculates the f1_score of the model using predicted and truth values
    
    Parameters:
        A (numpy matrix): predicted classification for each example
        Y (numpy matrix): true classification for each example
        thres (float) : threshold value, values above it are considered true
    
    Returns:
        (float): f1_score

    
`precision_score(A, Y, thres=0.5)`
:   This function calculates the precision of the model using predicted and truth values
    
    Parameters:
        A (numpy matrix): predicted classification for each example
        Y (numpy matrix): true classification for each example
        thres (float): threshold value, values above it are considered true
    
    Returns:
        (float): precision

    
`print_conf_mat(A, Y, thres=0.5)`
:   This function prints the confusion matrix from predicted and truth matrices
    
    Parameters:
        A (numpy matrix): predicted classification for each example
        Y (numpy matrix): true classification for each example
        thres (float) : threshold value, values above it are considered true

    
`recall_score(A, Y, thres=0.5)`
:   This function calculates the recall of the model using predicted and truth values
    
    Parameters:
        A (numpy matrix): predicted classification for each example
        Y (numpy matrix): true classification for each example
        thres (float): threshold value, values above it are considered true
    
    Returns:
        (float): recall