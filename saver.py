import pickle


def save(filename,model):
    pickle.dump(model, open(filename, 'wb'))

def retrive(filename):
    return pickle.load(open(filename, 'rb'))

