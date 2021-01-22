Module frame
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
            dictionary contains the updated parameters