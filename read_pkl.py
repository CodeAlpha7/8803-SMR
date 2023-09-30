import pickle

with open('pickled_data/saved_weight.pickle', 'rb') as f:
    content = pickle.load(f)
    print(content)