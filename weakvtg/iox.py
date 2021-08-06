import pickle


def load_json(filename):
    with open(filename, "r") as f:
        data = filename.load(f)
    return data


def load_pickle(filename, decompress=True):
    with open(filename, "rb") as f:
        if decompress:
            data = pickle.load(f)
        else:
            data = f.read()
    return data
