import pickle


def save(filename,model):
     """ This function saves the model ( all of it's parameters, weights, losses ,... ) in an external file 

    Parameters:
    filename (string): Path of file to be saved in, filename must be ".sav" type.
    model : Object "model" required to be saved.

    Returns:
    None

   """
    pickle.dump(model, open(filename, 'wb'))

def retrive(filename):
     """ This function loads a saved model ( all of it's parameters, weights, losses, ...) from external file

    Parameters:
    filename (string): Path of file to be retrived, filename must be ".sav" type.

    Returns:
    model : Object of model that saved before as whole with its structure(Layers,Nodes,Activation functions)

   """
    return pickle.load(open(filename, 'rb'))

