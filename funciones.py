"""
@author: Andres Fernando Guaca
"""

import pickle

def save_list(lst, filename):
    with open(filename, "w") as f:
        for coin in lst:
            f.write("{0}\n".format(coin))


def load_list(filename):
    lst = []
    with open(filename, "r") as f:
        for line in f:
            lst.append(line.strip())
    return lst

def load_objeto(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_object(file, objeto):
    with open(file, 'wb') as output:
        pickle.dump(objeto, output, pickle.HIGHEST_PROTOCOL)
        