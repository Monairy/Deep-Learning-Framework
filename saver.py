import pickle


def save(filename,model):
     """ This function saves the model ( all of it's parameters, weights, losses ,... ) in an external file 

    Parameters:
    filename (path?!): Path of file to be saved in.
    model : Object "model" required to be saved.

    Returns:
    None

   """
    pickle.dump(model, open(filename, 'wb'))

def retrive(filename):
     """ This function loads a saved model ( all of it's parameters, weights, losses, ...) from external file

    Parameters:
    filename (path?!): Path of file to be retrived

    Returns:
    model : ?!

   """
    return pickle.load(open(filename, 'rb'))

