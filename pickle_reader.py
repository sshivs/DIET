import pickle

def read_pickle_file(fname):
    pickle_data = pickle.load(open(fname, 'rb'))
    return pickle_data
